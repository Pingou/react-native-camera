package org.reactnative.camera.events;

import androidx.core.util.Pools;
import android.util.SparseArray;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import android.util.Log;
import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.uimanager.events.Event;
import com.facebook.react.uimanager.events.RCTEventEmitter;
import com.google.android.cameraview.CameraView;
import com.google.android.gms.vision.text.Line;
import com.google.android.gms.vision.text.Text;
import com.google.android.gms.vision.text.TextBlock;
import org.reactnative.camera.CameraViewManager;
import org.reactnative.camera.utils.ImageDimensions;
import org.reactnative.facedetector.FaceDetectorUtils;


public class TFProcessedEvent extends Event<TFProcessedEvent> {

  private static final Pools.SynchronizedPool<TFProcessedEvent> EVENTS_POOL =
      new Pools.SynchronizedPool<>(3);


 
  private WritableArray mData;


  private TFProcessedEvent() {}

  public static TFProcessedEvent obtain(
      int viewTag,
      WritableArray list) {
    TFProcessedEvent event = EVENTS_POOL.acquire();
    if (event == null) {
      event = new TFProcessedEvent();
    }
    event.init(viewTag, list);
    return event;
  }

  private void init(
      int viewTag,
      WritableArray list) {
    super.init(viewTag);
    mData = list;
    
  }

  @Override
  public String getEventName() {
    return CameraViewManager.Events.EVENT_ON_TF_PROCESSED.toString();
  }

  @Override
  public void dispatch(RCTEventEmitter rctEventEmitter) {

    try {
      rctEventEmitter.receiveEvent(getViewTag(), getEventName(), serializeEventData());
    }
    catch(Exception e) {
      Log.e("TFProcessedEvent", "error");
    }
  }

  private WritableMap serializeEventData() {

    WritableMap event = Arguments.createMap();
    event.putString("type", "textBlock");
    event.putArray("data", mData);
    event.putInt("target", getViewTag());
    return event;
  }

}
