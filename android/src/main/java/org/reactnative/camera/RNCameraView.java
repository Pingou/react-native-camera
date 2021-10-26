package org.reactnative.camera;

import android.util.Log;
import android.view.TextureView;
import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.media.CamcorderProfile;
import android.media.MediaActionSound;
import android.os.Build;
import androidx.core.content.ContextCompat;
import java.nio.ByteBuffer;
import android.view.View;
import android.os.AsyncTask;
import com.facebook.react.bridge.*;
import com.facebook.react.uimanager.ThemedReactContext;
import com.google.android.cameraview.CameraView;
import com.google.zxing.BarcodeFormat;
import com.google.zxing.DecodeHintType;
import com.google.zxing.MultiFormatReader;
import com.google.zxing.Result;
import org.reactnative.barcodedetector.RNBarcodeDetector;
import org.reactnative.camera.tasks.*;
import org.reactnative.camera.tflite.Classifier;

import org.reactnative.camera.tflite.ImageUtils;
import org.reactnative.camera.utils.ImageDimensions;
import org.reactnative.camera.utils.RNFileUtils;
import org.reactnative.facedetector.RNFaceDetector;
import android.content.res.AssetFileDescriptor;
import java.io.FileInputStream;
import java.nio.channels.FileChannel;
import java.nio.MappedByteBuffer;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;


import org.reactnative.camera.tflite.Classifier.Model;
import android.os.Handler;
import android.os.HandlerThread;
import android.media.Image;
import android.media.Image.Plane;

public class RNCameraView extends CameraView implements LifecycleEventListener, BarCodeScannerAsyncTaskDelegate, FaceDetectorAsyncTaskDelegate,
    BarcodeDetectorAsyncTaskDelegate, TextRecognizerAsyncTaskDelegate, PictureSavedDelegate, ModelProcessorAsyncTaskDelegate {
  private ThemedReactContext mThemedReactContext;
  private Queue<Promise> mPictureTakenPromises = new ConcurrentLinkedQueue<>();
  private Map<Promise, ReadableMap> mPictureTakenOptions = new ConcurrentHashMap<>();
  private Map<Promise, File> mPictureTakenDirectories = new ConcurrentHashMap<>();
  private Promise mVideoRecordedPromise;
  private List<String> mBarCodeTypes = null;
  private Boolean mPlaySoundOnCapture = false;

  private boolean mIsPaused = false;
  private boolean mIsNew = true;
  private boolean invertImageData = false;
  private Boolean mIsRecording = false;
  private Boolean mIsRecordingInterrupted = false;

  // Concurrency lock for scanners to avoid flooding the runtime
  public volatile boolean barCodeScannerTaskLock = false;
  public volatile boolean faceDetectorTaskLock = false;
  public volatile boolean googleBarcodeDetectorTaskLock = false;
  public volatile boolean textRecognizerTaskLock = false;
  public volatile boolean modelProcessorTaskLock = false;

  // Scanning-related properties
  private MultiFormatReader mMultiFormatReader;
  private RNFaceDetector mFaceDetector;
  private RNBarcodeDetector mGoogleBarcodeDetector;
  private String mModelFile;
  private Interpreter mModelProcessor;
  private int mModelMaxFreqms;
  private ByteBuffer mModelInput;
  private int[] mModelViewBuf;
  private int mModelImageDimX;
  private int mModelImageDimY;
  private int mModelOutputDim;
  private ByteBuffer mModelOutput;
  private boolean mShouldDetectFaces = false;
  private boolean mShouldGoogleDetectBarcodes = false;
  private boolean mShouldScanBarCodes = false;
  private boolean mShouldRecognizeText = false;
  private boolean mShouldProcessModel = false;
  private int mFaceDetectorMode = RNFaceDetector.FAST_MODE;
  private int mFaceDetectionLandmarks = RNFaceDetector.NO_LANDMARKS;
  private int mFaceDetectionClassifications = RNFaceDetector.NO_CLASSIFICATIONS;
  private int mGoogleVisionBarCodeType = RNBarcodeDetector.ALL_FORMATS;
  private int mGoogleVisionBarCodeMode = RNBarcodeDetector.NORMAL_MODE;
  private boolean mTrackingEnabled = true;
  private int mPaddingX;
  private int mPaddingY;




  private Classifier classifier = null;
  private Model model = Model.FLOAT_EFFICIENTNET;
  private Bitmap rgbFrameBitmap = null;
  /** Input image size of the model along x axis. */
  private int imageSizeX;
  /** Input image size of the model along y axis. */
  private int imageSizeY;
  protected int previewWidth = 0;
  protected int previewHeight = 0;
  protected boolean isProcessingFrame = false;
  private byte[][] yuvBytes = new byte[3][];
  private int[] rgbBytes = null;
  private int yRowStride;
  private Image previewImage = null;
  private Runnable imageConverter;
  private Runnable postInferenceCallback;



  private Handler handler;
  private HandlerThread handlerThread;

  public RNCameraView(ThemedReactContext themedReactContext) {
    super(themedReactContext, true);
    mThemedReactContext = themedReactContext;
    themedReactContext.addLifecycleEventListener(this);

   // handlerThread = new HandlerThread("inference");
   // handlerThread.start();
   // handler = new Handler(handlerThread.getLooper());

    try {
      classifier = Classifier.create(themedReactContext.getCurrentActivity(), model, 1);
    } catch (IOException e) {
        Log.v("pingou error",  "Failed to create classifier.");
    }
    Log.v("pingou",  "Classifier created.");
    // Updates the input image size.
    imageSizeX = classifier.getImageSizeX();
    imageSizeY = classifier.getImageSizeY();


    setScanning(true);
    addCallback(new Callback() {
      @Override
      public void onCameraOpened(CameraView cameraView) {
        RNCameraViewHelper.emitCameraReadyEvent(cameraView);
      }

      @Override
      public void onMountError(CameraView cameraView) {
        RNCameraViewHelper.emitMountErrorEvent(cameraView, "Camera view threw an error - component could not be rendered.");
      }

      @Override
      public void onPictureTaken(CameraView cameraView, final byte[] data, int deviceOrientation) {
        Promise promise = mPictureTakenPromises.poll();
        ReadableMap options = mPictureTakenOptions.remove(promise);
        if (options.hasKey("fastMode") && options.getBoolean("fastMode")) {
            promise.resolve(null);
        }
        final File cacheDirectory = mPictureTakenDirectories.remove(promise);
        if(Build.VERSION.SDK_INT >= 11/*HONEYCOMB*/) {
          new ResolveTakenPictureAsyncTask(data, promise, options, cacheDirectory, deviceOrientation, RNCameraView.this)
                  .executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
        } else {
          new ResolveTakenPictureAsyncTask(data, promise, options, cacheDirectory, deviceOrientation, RNCameraView.this)
                  .execute();
        }
        RNCameraViewHelper.emitPictureTakenEvent(cameraView);
      }

      @Override
      public void onVideoRecorded(CameraView cameraView, String path, int videoOrientation, int deviceOrientation) {
        if (mVideoRecordedPromise != null) {
          if (path != null) {
            WritableMap result = Arguments.createMap();
            result.putBoolean("isRecordingInterrupted", mIsRecordingInterrupted);
            result.putInt("videoOrientation", videoOrientation);
            result.putInt("deviceOrientation", deviceOrientation);
            result.putString("uri", RNFileUtils.uriFromFile(new File(path)).toString());
            mVideoRecordedPromise.resolve(result);
          } else {
            mVideoRecordedPromise.reject("E_RECORDING", "Couldn't stop recording - there is none in progress");
          }
          mIsRecording = false;
          mIsRecordingInterrupted = false;
          mVideoRecordedPromise = null;
        }
      }

      @Override
      public void onFramePreview(CameraView cameraView, Image image, final byte[] data, int width, int height, int rotation) {
        int correctRotation = RNCameraViewHelper.getCorrectCameraRotation(rotation, getFacing(), getCameraOrientation());
        boolean willCallBarCodeTask = mShouldScanBarCodes && !barCodeScannerTaskLock && cameraView instanceof BarCodeScannerAsyncTaskDelegate;
        boolean willCallFaceTask = mShouldDetectFaces && !faceDetectorTaskLock && cameraView instanceof FaceDetectorAsyncTaskDelegate;
        boolean willCallGoogleBarcodeTask = mShouldGoogleDetectBarcodes && !googleBarcodeDetectorTaskLock && cameraView instanceof BarcodeDetectorAsyncTaskDelegate;
        boolean willCallTextTask = mShouldRecognizeText && !textRecognizerTaskLock && cameraView instanceof TextRecognizerAsyncTaskDelegate;
	    boolean willCallModelTask = mShouldProcessModel && !modelProcessorTaskLock && cameraView instanceof ModelProcessorAsyncTaskDelegate;


	  /*  if (!willCallBarCodeTask && !willCallFaceTask && !willCallGoogleBarcodeTask && !willCallTextTask && !willCallModelTask) {
          return;
        }

        if (data.length < (1.5 * width * height)) {
            return;
        }*/
        if (data.length < (1.5 * width * height)) {

          if (image != null)
            image.close();
          return;
        }


        if (willCallBarCodeTask) {
          barCodeScannerTaskLock = true;
          BarCodeScannerAsyncTaskDelegate delegate = (BarCodeScannerAsyncTaskDelegate) cameraView;
          new BarCodeScannerAsyncTask(delegate, mMultiFormatReader, data, width, height).execute();
        }

        if (willCallFaceTask) {
          faceDetectorTaskLock = true;
          FaceDetectorAsyncTaskDelegate delegate = (FaceDetectorAsyncTaskDelegate) cameraView;
          new FaceDetectorAsyncTask(delegate, mFaceDetector, data, width, height, correctRotation, getResources().getDisplayMetrics().density, getFacing(), getWidth(), getHeight(), mPaddingX, mPaddingY).execute();
        }

        if (willCallGoogleBarcodeTask) {
          googleBarcodeDetectorTaskLock = true;
          if (mGoogleVisionBarCodeMode == RNBarcodeDetector.NORMAL_MODE) {
            invertImageData = false;
          } else if (mGoogleVisionBarCodeMode == RNBarcodeDetector.ALTERNATE_MODE) {
            invertImageData = !invertImageData;
          } else if (mGoogleVisionBarCodeMode == RNBarcodeDetector.INVERTED_MODE) {
            invertImageData = true;
          }
          if (invertImageData) {
            for (int y = 0; y < data.length; y++) {
              data[y] = (byte) ~data[y];
            }
          }
          BarcodeDetectorAsyncTaskDelegate delegate = (BarcodeDetectorAsyncTaskDelegate) cameraView;
          new BarcodeDetectorAsyncTask(delegate, mGoogleBarcodeDetector, data, width, height, correctRotation, getResources().getDisplayMetrics().density, getFacing(), getWidth(), getHeight(), mPaddingX, mPaddingY).execute();
        }

        if (willCallTextTask) {
          textRecognizerTaskLock = true;
          TextRecognizerAsyncTaskDelegate delegate = (TextRecognizerAsyncTaskDelegate) cameraView;
          new TextRecognizerAsyncTask(delegate, mThemedReactContext, data, width, height, correctRotation, getResources().getDisplayMetrics().density, getFacing(), getWidth(), getHeight(), mPaddingX, mPaddingY).execute();
        }

	    /*if (willCallModelTask) {
          modelProcessorTaskLock = true;
          getImageData((TextureView) cameraView.getView());
          ModelProcessorAsyncTaskDelegate delegate = (ModelProcessorAsyncTaskDelegate) cameraView;
          new ModelProcessorAsyncTask(delegate, mModelProcessor, mModelInput, mModelOutput, mModelMaxFreqms, width, height, correctRotation).execute();
        }*/

        preprocessImage(image, data, width, height, rotation);


      }
    });
  }


  private void preprocessImage(Image image, final byte[] data, int width, int height, int rotation) {

    if (isProcessingFrame) {
        Log.d("log", "Dropping frame!");
        if (image != null)
          image.close();
        return;
    }

    isProcessingFrame = true;
    //camera api 1
    if (image == null) {
      try {
        // Initialize the storage bitmaps once when the resolution is known.
        if (rgbBytes == null) {
          previewHeight = height;
          previewWidth = width;
          rgbBytes = new int[previewWidth * previewHeight];
          rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        }
      } catch (final Exception e) {
        Log.d("error", "Initialize the storage bitmaps ");

        isProcessingFrame = false;
        return;
      }


      yuvBytes[0] = data;
      yRowStride = previewWidth;

      imageConverter =
              new Runnable() {
                @Override
                public void run() {

                  try {
                    ImageUtils.convertYUV420SPToARGB8888(data, previewWidth, previewHeight, rgbBytes);
                  }
                  catch (Exception e) {
                    Log.d("error", "convertYUV420SPToARGB8888 ");
                  }
                }
              };


    }
    //camera2
    else {
      if (rgbBytes == null) {
        previewHeight = height;
        previewWidth = width;
        rgbBytes = new int[previewWidth * previewHeight];
      }
      try {

        previewImage = image;
        final Plane[] planes = image.getPlanes();
        fillBytes(planes, yuvBytes);
        yRowStride = planes[0].getRowStride();
        final int uvRowStride = planes[1].getRowStride();
        final int uvPixelStride = planes[1].getPixelStride();

        imageConverter =
                new Runnable() {
                  @Override
                  public void run() {
                    ImageUtils.convertYUV420ToARGB8888(
                            yuvBytes[0],
                            yuvBytes[1],
                            yuvBytes[2],
                            previewWidth,
                            previewHeight,
                            yRowStride,
                            uvRowStride,
                            uvPixelStride,
                            rgbBytes);
                  }
                };
      } catch (final Exception e) {
        Log.d("pingou error", "Exception camera2!");
        isProcessingFrame = false;
        image.close();
        previewImage = null;
        return;
      }
    }



      processImage(rotation);
  }


  protected void fillBytes(final Plane[] planes, final byte[][] yuvBytes) {
    // Because of the variable row stride it's not possible to know in
    // advance the actual necessary dimensions of the yuv planes.
    for (int i = 0; i < planes.length; ++i) {
      final ByteBuffer buffer = planes[i].getBuffer();
      if (yuvBytes[i] == null) {
//        Log.d("Initializing buffer %d at size %d", i, buffer.capacity());
        yuvBytes[i] = new byte[buffer.capacity()];
      }
      buffer.get(yuvBytes[i]);
    }
  }
  protected void readyForNextImage() {
    if (postInferenceCallback != null) {
      postInferenceCallback.run();
    }
  }


  protected int[] getRgbBytes() {
    imageConverter.run();
    return rgbBytes;
  }

  protected int getLuminanceStride() {
    return yRowStride;
  }

  protected byte[] getLuminance() {
    return yuvBytes[0];
  }

  protected void processImage(final int orientation) {
    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
    final int cropSize = Math.min(previewWidth, previewHeight);

    AsyncTask.execute(new Runnable() {
      @Override
      public void run() {
        if (classifier != null) {

          try {
            final List<Classifier.Recognition> results =
                    classifier.recognizeImage(rgbFrameBitmap, orientation);

            if (previewImage != null) {
              previewImage.close();
              previewImage = null;
            }
            final WritableArray resultList = Arguments.createArray();
            for (Classifier.Recognition result : results) {
              WritableMap serializedItem = Arguments.createMap();

              //serializedItem.putString("label", result.getTitle());
              serializedItem.putDouble("score", result.getConfidence());
              serializedItem.putInt("pos", result.getPosition());
              resultList.pushMap(serializedItem);
            }


            mThemedReactContext.getCurrentActivity().runOnUiThread(new Runnable() {
              @Override
              public void run() {
                Log.v("pingou", "got results." + String.valueOf(results.size()));
                RNCameraViewHelper.emitTFProcessedEvent(RNCameraView.this, resultList);

              }
            });
          }
          catch (Exception e) {

          }
        }
        isProcessingFrame = false;
      }
    });

    /*
    runInBackground(
            new Runnable() {
              @Override
              public void run() {
                if (classifier != null) {
                  final List<Classifier.Recognition> results =
                          classifier.recognizeImage(rgbFrameBitmap, orientation);
                }
                isProcessingFrame = false;
                // readyForNextImage();
              }


            });*/
  }

    protected synchronized void runInBackground(final Runnable r) {
      if (handler != null) {
        handler.post(r);
      }
    }

  /*
  private void getImageData(TextureView view) {
    Bitmap bitmap = view.getBitmap(mModelImageDimX, mModelImageDimY);
    if (bitmap == null) {
      return;
    }
    mModelInput.rewind();
    bitmap.getPixels(mModelViewBuf, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    int pixel = 0;
    for (int i = 0; i < mModelImageDimX; ++i) {
      for (int j = 0; j < mModelImageDimY; ++j) {
        final int val = mModelViewBuf[pixel++];
        mModelInput.put((byte) ((val >> 16) & 0xFF));
        mModelInput.put((byte) ((val >> 8) & 0xFF));
        mModelInput.put((byte) (val & 0xFF));
      }
    }
  }
*/
  @Override
  protected void onLayout(boolean changed, int left, int top, int right, int bottom) {
    View preview = getView();
    if (null == preview) {
      return;
    }
    float width = right - left;
    float height = bottom - top;
    float ratio = getAspectRatio().toFloat();
    int orientation = getResources().getConfiguration().orientation;
    int correctHeight;
    int correctWidth;
    this.setBackgroundColor(Color.BLACK);
    if (orientation == android.content.res.Configuration.ORIENTATION_LANDSCAPE) {
      if (ratio * height < width) {
        correctHeight = (int) (width / ratio);
        correctWidth = (int) width;
      } else {
        correctWidth = (int) (height * ratio);
        correctHeight = (int) height;
      }
    } else {
      if (ratio * width > height) {
        correctHeight = (int) (width * ratio);
        correctWidth = (int) width;
      } else {
        correctWidth = (int) (height / ratio);
        correctHeight = (int) height;
      }
    }
    int paddingX = (int) ((width - correctWidth) / 2);
    int paddingY = (int) ((height - correctHeight) / 2);
    mPaddingX = paddingX;
    mPaddingY = paddingY;
    preview.layout(paddingX, paddingY, correctWidth + paddingX, correctHeight + paddingY);
  }

  @SuppressLint("all")
  @Override
  public void requestLayout() {
    // React handles this for us, so we don't need to call super.requestLayout();
  }

  public void setBarCodeTypes(List<String> barCodeTypes) {
    mBarCodeTypes = barCodeTypes;
    initBarcodeReader();
  }

  public void setPlaySoundOnCapture(Boolean playSoundOnCapture) {
    mPlaySoundOnCapture = playSoundOnCapture;
  }

  public void takePicture(ReadableMap options, final Promise promise, File cacheDirectory) {
    mPictureTakenPromises.add(promise);
    mPictureTakenOptions.put(promise, options);
    mPictureTakenDirectories.put(promise, cacheDirectory);
    if (mPlaySoundOnCapture) {
      MediaActionSound sound = new MediaActionSound();
      sound.play(MediaActionSound.SHUTTER_CLICK);
    }
    try {
      super.takePicture(options);
    } catch (Exception e) {
      mPictureTakenPromises.remove(promise);
      mPictureTakenOptions.remove(promise);
      mPictureTakenDirectories.remove(promise);
      throw e;
    }
  }

  @Override
  public void onPictureSaved(WritableMap response) {
    RNCameraViewHelper.emitPictureSavedEvent(this, response);
  }

  public void record(ReadableMap options, final Promise promise, File cacheDirectory) {
    try {
      String path = options.hasKey("path") ? options.getString("path") : RNFileUtils.getOutputFilePath(cacheDirectory, ".mp4");
      int maxDuration = options.hasKey("maxDuration") ? options.getInt("maxDuration") : -1;
      int maxFileSize = options.hasKey("maxFileSize") ? options.getInt("maxFileSize") : -1;

      CamcorderProfile profile = CamcorderProfile.get(CamcorderProfile.QUALITY_HIGH);
      if (options.hasKey("quality")) {
        profile = RNCameraViewHelper.getCamcorderProfile(options.getInt("quality"));
      }
      if (options.hasKey("videoBitrate")) {
        profile.videoBitRate = options.getInt("videoBitrate");
      }

      boolean recordAudio = true;
      if (options.hasKey("mute")) {
        recordAudio = !options.getBoolean("mute");
      }

      int orientation = Constants.ORIENTATION_AUTO;
      if (options.hasKey("orientation")) {
        orientation = options.getInt("orientation");
      }

      if (super.record(path, maxDuration * 1000, maxFileSize, recordAudio, profile, orientation)) {
        mIsRecording = true;
        mVideoRecordedPromise = promise;
      } else {
        promise.reject("E_RECORDING_FAILED", "Starting video recording failed. Another recording might be in progress.");
      }
    } catch (IOException e) {
      promise.reject("E_RECORDING_FAILED", "Starting video recording failed - could not create video file.");
    }
  }

  /**
   * Initialize the barcode decoder.
   * Supports all iOS codes except [code138, code39mod43, itf14]
   * Additionally supports [codabar, code128, maxicode, rss14, rssexpanded, upc_a, upc_ean]
   */
  private void initBarcodeReader() {
    mMultiFormatReader = new MultiFormatReader();
    EnumMap<DecodeHintType, Object> hints = new EnumMap<>(DecodeHintType.class);
    EnumSet<BarcodeFormat> decodeFormats = EnumSet.noneOf(BarcodeFormat.class);

    if (mBarCodeTypes != null) {
      for (String code : mBarCodeTypes) {
        String formatString = (String) CameraModule.VALID_BARCODE_TYPES.get(code);
        if (formatString != null) {
          decodeFormats.add(BarcodeFormat.valueOf(formatString));
        }
      }
    }

    hints.put(DecodeHintType.POSSIBLE_FORMATS, decodeFormats);
    mMultiFormatReader.setHints(hints);
  }

  public void setShouldScanBarCodes(boolean shouldScanBarCodes) {
    if (shouldScanBarCodes && mMultiFormatReader == null) {
      initBarcodeReader();
    }
    this.mShouldScanBarCodes = shouldScanBarCodes;
    setScanning(mShouldDetectFaces || mShouldGoogleDetectBarcodes || mShouldScanBarCodes || mShouldRecognizeText || mShouldProcessModel);
  }

  public void onBarCodeRead(Result barCode, int width, int height) {
    String barCodeType = barCode.getBarcodeFormat().toString();
    if (!mShouldScanBarCodes || !mBarCodeTypes.contains(barCodeType)) {
      return;
    }

    RNCameraViewHelper.emitBarCodeReadEvent(this, barCode,  width,  height);
  }

  public void onBarCodeScanningTaskCompleted() {
    barCodeScannerTaskLock = false;
    if(mMultiFormatReader != null) {
      mMultiFormatReader.reset();
    }
  }

  /**
   * Initial setup of the face detector
   */
  private void setupFaceDetector() {
    mFaceDetector = new RNFaceDetector(mThemedReactContext);
    mFaceDetector.setMode(mFaceDetectorMode);
    mFaceDetector.setLandmarkType(mFaceDetectionLandmarks);
    mFaceDetector.setClassificationType(mFaceDetectionClassifications);
    mFaceDetector.setTracking(mTrackingEnabled);
  }

  public void setFaceDetectionLandmarks(int landmarks) {
    mFaceDetectionLandmarks = landmarks;
    if (mFaceDetector != null) {
      mFaceDetector.setLandmarkType(landmarks);
    }
  }

  public void setFaceDetectionClassifications(int classifications) {
    mFaceDetectionClassifications = classifications;
    if (mFaceDetector != null) {
      mFaceDetector.setClassificationType(classifications);
    }
  }

  public void setFaceDetectionMode(int mode) {
    mFaceDetectorMode = mode;
    if (mFaceDetector != null) {
      mFaceDetector.setMode(mode);
    }
  }

  public void setTracking(boolean trackingEnabled) {
    mTrackingEnabled = trackingEnabled;
    if (mFaceDetector != null) {
      mFaceDetector.setTracking(trackingEnabled);
    }
  }

  public void setShouldDetectFaces(boolean shouldDetectFaces) {
    if (shouldDetectFaces && mFaceDetector == null) {
      setupFaceDetector();
    }
    this.mShouldDetectFaces = shouldDetectFaces;
    setScanning(mShouldDetectFaces || mShouldGoogleDetectBarcodes || mShouldScanBarCodes || mShouldRecognizeText || mShouldProcessModel);
  }

  public void onFacesDetected(WritableArray data) {
    if (!mShouldDetectFaces) {
      return;
    }

    RNCameraViewHelper.emitFacesDetectedEvent(this, data);
  }

  public void onFaceDetectionError(RNFaceDetector faceDetector) {
    if (!mShouldDetectFaces) {
      return;
    }

    RNCameraViewHelper.emitFaceDetectionErrorEvent(this, faceDetector);
  }

  @Override
  public void onFaceDetectingTaskCompleted() {
    faceDetectorTaskLock = false;
  }

  /**
   * Initial setup of the barcode detector
   */
  private void setupBarcodeDetector() {
    mGoogleBarcodeDetector = new RNBarcodeDetector(mThemedReactContext);
    mGoogleBarcodeDetector.setBarcodeType(mGoogleVisionBarCodeType);
  }

  public void setShouldGoogleDetectBarcodes(boolean shouldDetectBarcodes) {
    if (shouldDetectBarcodes && mGoogleBarcodeDetector == null) {
      setupBarcodeDetector();
    }
    this.mShouldGoogleDetectBarcodes = shouldDetectBarcodes;
    setScanning(mShouldDetectFaces || mShouldGoogleDetectBarcodes || mShouldScanBarCodes || mShouldRecognizeText || mShouldProcessModel);
  }

  private MappedByteBuffer loadModelFile() throws IOException {
    AssetFileDescriptor fileDescriptor = mThemedReactContext.getAssets().openFd(mModelFile);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /*
  private void setupModelProcessor() {
    try {
      mModelProcessor = new Interpreter(loadModelFile());
      mModelInput = ByteBuffer.allocateDirect(mModelImageDimX * mModelImageDimY * 3);
      mModelViewBuf = new int[mModelImageDimX * mModelImageDimY];
      mModelOutput = ByteBuffer.allocateDirect(mModelOutputDim);
    } catch(Exception e) {}
  }
*/
  public void setGoogleVisionBarcodeType(int barcodeType) {
    mGoogleVisionBarCodeType = barcodeType;
    if (mGoogleBarcodeDetector != null) {
      mGoogleBarcodeDetector.setBarcodeType(barcodeType);
    }
  }

  public void setGoogleVisionBarcodeMode(int barcodeMode) {
    mGoogleVisionBarCodeMode = barcodeMode;
  }

  public void onBarcodesDetected(WritableArray barcodesDetected) {
    if (!mShouldGoogleDetectBarcodes) {
      return;
    }
    RNCameraViewHelper.emitBarcodesDetectedEvent(this, barcodesDetected);
  }

  public void onBarcodeDetectionError(RNBarcodeDetector barcodeDetector) {
    if (!mShouldGoogleDetectBarcodes) {
      return;
    }

    RNCameraViewHelper.emitBarcodeDetectionErrorEvent(this, barcodeDetector);
  }

  @Override
  public void onBarcodeDetectingTaskCompleted() {
    googleBarcodeDetectorTaskLock = false;
  }

  /**
   *
   * Text recognition
   */

  public void setShouldRecognizeText(boolean shouldRecognizeText) {
    this.mShouldRecognizeText = shouldRecognizeText;
    setScanning(mShouldDetectFaces || mShouldGoogleDetectBarcodes || mShouldScanBarCodes || mShouldRecognizeText);
  }

  /*
  public void setModelFile(String modelFile, int inputDimX, int inputDimY, int outputDim, int freqms) {
    this.mModelFile = modelFile;
    this.mModelImageDimX = inputDimX;
    this.mModelImageDimY = inputDimY;
    this.mModelOutputDim = outputDim;
    this.mModelMaxFreqms = freqms;
    boolean shouldProcessModel = (modelFile != null);
    if (shouldProcessModel && mModelProcessor == null) {
      setupModelProcessor();
    }
    this.mShouldProcessModel = shouldProcessModel;
    setScanning(mShouldDetectFaces || mShouldGoogleDetectBarcodes || mShouldScanBarCodes || mShouldRecognizeText || mShouldProcessModel);
  }
  */

  public void onTextRecognized(WritableArray serializedData) {
    if (!mShouldRecognizeText) {
      return;
    }

    RNCameraViewHelper.emitTextRecognizedEvent(this, serializedData);
  }

  @Override
  public void onModelProcessed(ByteBuffer data, int sourceWidth, int sourceHeight, int sourceRotation) {
    if (!mShouldProcessModel) {
      return;
    }

    ByteBuffer dataDetected = data == null ? ByteBuffer.allocate(0) : data;
    ImageDimensions dimensions = new ImageDimensions(sourceWidth, sourceHeight, sourceRotation, getFacing());

    RNCameraViewHelper.emitModelProcessedEvent(this, dataDetected, dimensions);
  }

  @Override
  public void onTextRecognizerTaskCompleted() {
    textRecognizerTaskLock = false;
  }

  @Override
  public void onModelProcessorTaskCompleted() {
    modelProcessorTaskLock = false;
  }


  /**
  *
  * End Text Recognition */

  @Override
  public void onHostResume() {
    if (hasCameraPermissions()) {
      if ((mIsPaused && !isCameraOpened()) || mIsNew) {
        mIsPaused = false;
        mIsNew = false;
        start();
      }
    } else {
      RNCameraViewHelper.emitMountErrorEvent(this, "Camera permissions not granted - component could not be rendered.");
    }
  }

  @Override
  public void onHostPause() {
    if (mIsRecording) {
      mIsRecordingInterrupted = true;
    }
    if (!mIsPaused && isCameraOpened()) {
      mIsPaused = true;
      stop();
    }
  }

  @Override
  public void onHostDestroy() {
    if (mFaceDetector != null) {
      mFaceDetector.release();
    }
    if (mGoogleBarcodeDetector != null) {
      mGoogleBarcodeDetector.release();
    }
    if (mModelProcessor != null) {
      mModelProcessor.close();
    }

    mMultiFormatReader = null;
    stop();
    mThemedReactContext.removeLifecycleEventListener(this);
  }

  private boolean hasCameraPermissions() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      int result = ContextCompat.checkSelfPermission(getContext(), Manifest.permission.CAMERA);
      return result == PackageManager.PERMISSION_GRANTED;
    } else {
      return true;
    }
  }
}
