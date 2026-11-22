package org.echogesture.app;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.os.*;
import android.speech.tts.TextToSpeech;
import android.speech.tts.Voice;
import android.speech.tts.UtteranceProgressListener;
import android.view.*;
import android.widget.*;
import androidx.activity.ComponentActivity;
import androidx.camera.core.*;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mediapipe.framework.image.*;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.holisticlandmarker.*;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import org.json.*;
import java.util.*;
import java.util.concurrent.*;

/** Offline camera prototype. User starts/stops each isolated sign and confirms.
 * No video file, internet permission, analytics, or inferred sentence rewriting.
 */
public class MainActivity extends ComponentActivity {
    private final ExecutorService worker=Executors.newSingleThreadExecutor();
    private HolisticLandmarker holistic;
    private SignEngine recognizer;
    private PreviewView preview;
    private TextView status, caption, timing, speechStatus, captionHud;
    private Button recordButton, benchmarkButton, speakButton;
    private TextToSpeech speech;
    private boolean speechReady=false;
    private final ArrayList<Button> choices=new ArrayList<>();
    private final ArrayList<String> confirmed=new ArrayList<>();
    private final ArrayList<float[][]> sequence=new ArrayList<>();
    private final ArrayList<Double> frameTimes=new ArrayList<>(), classifierTimes=new ArrayList<>();
    private final JSONArray samples=new JSONArray();
    private volatile boolean recording=false, benchmarking=false;
    private boolean ready=false, handsSeen=false;
    private long recordingStart=0, benchmarkStart=0, lastTimestamp=0;
    private String[] pending=null;
    private volatile boolean destroyed=false;
    private ProcessCameraProvider cameraProvider;

    @Override public void onCreate(Bundle saved) {
        super.onCreate(saved);
        LinearLayout root=new LinearLayout(this); root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(24,24,24,24); root.setBackgroundColor(Color.rgb(16,27,27));
        root.setOnApplyWindowInsetsListener((v,insets)-> {
            v.setPadding(24,insets.getSystemWindowInsetTop()+16,24,insets.getSystemWindowInsetBottom()+16);
            return insets;
        });
        ScrollView scroll=new ScrollView(this); scroll.addView(root); setContentView(scroll);
        TextView title=new TextView(this);title.setText("EchoGesture");title.setTextSize(30);title.setTextColor(Color.rgb(169,240,196)); root.addView(title);
        TextView privacy=new TextView(this);privacy.setText("On this phone. Review each word before adding it.\nPrototype for isolated ASL words.");privacy.setTextColor(Color.LTGRAY);root.addView(privacy);
        FrameLayout cameraFrame=new FrameLayout(this);root.addView(cameraFrame,new LinearLayout.LayoutParams(-1,520));
        preview=new PreviewView(this);cameraFrame.addView(preview,new FrameLayout.LayoutParams(-1,-1));
        captionHud=new TextView(this);captionHud.setTextColor(Color.WHITE);captionHud.setTextSize(22);captionHud.setPadding(16,12,16,12);
        captionHud.setBackgroundColor(0xDD10221A);captionHud.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);
        FrameLayout.LayoutParams hudLayout=new FrameLayout.LayoutParams(-1,-2,Gravity.BOTTOM);cameraFrame.addView(captionHud,hudLayout);
        status=new TextView(this);status.setText("Preparing local models…");status.setTextSize(18);status.setTextColor(Color.WHITE);root.addView(status);
        timing=new TextView(this);timing.setTextColor(Color.LTGRAY);root.addView(timing);
        recordButton=new Button(this);recordButton.setText("Start one sign");root.addView(recordButton);
        recordButton.setOnClickListener(v->worker.execute(()-> {
            if(!ready||benchmarking) return;
            if(recording) finishSign();
            else {
                sequence.clear();handsSeen=false;recording=true;recordingStart=SystemClock.elapsedRealtime();
                runOnUiThread(()-> {stopSpeech();clearChoices();status.setText("Sign one word, then tap Review.");recordButton.setText("Review this sign");});
            }
        }));
        LinearLayout options=new LinearLayout(this);options.setOrientation(LinearLayout.VERTICAL);root.addView(options);
        for(int i=0;i<5;i++) {
            final int rank=i;Button choice=new Button(this);choice.setVisibility(View.GONE);options.addView(choice);choices.add(choice);
            choice.setOnClickListener(v->{if(pending!=null&&rank<pending.length){confirmed.add(pending[rank]);clearChoices();refreshCaption();status.setText("Added. You can undo the last word.");}});
        }
        caption=new TextView(this);caption.setTextSize(25);caption.setTextColor(Color.WHITE);root.addView(caption);refreshCaption();
        LinearLayout edits=new LinearLayout(this);root.addView(edits);
        Button undo=new Button(this);undo.setText("Undo");edits.addView(undo);undo.setOnClickListener(v->{stopSpeech();if(!confirmed.isEmpty())confirmed.remove(confirmed.size()-1);refreshCaption();});
        Button clear=new Button(this);clear.setText("Clear");edits.addView(clear);clear.setOnClickListener(v->{stopSpeech();confirmed.clear();clearChoices();refreshCaption();status.setText("Caption cleared from memory.");});
        speakButton=new Button(this);speakButton.setText("Speak confirmed words");speakButton.setEnabled(false);root.addView(speakButton);
        speechStatus=new TextView(this);speechStatus.setTextColor(Color.LTGRAY);speechStatus.setText("Checking for an offline English voice…");root.addView(speechStatus);
        speakButton.setOnClickListener(v->{
            if(!speechReady||speech==null||confirmed.isEmpty()||recording||benchmarking)return;
            String text=String.join(" ",confirmed).replace('_',' ').toLowerCase(Locale.US);
            if(text.length()>TextToSpeech.getMaxSpeechInputLength()){
                speechStatus.setText("This caption is too long to speak at once. Clear it and speak a shorter phrase.");return;
            }
            if(speech.speak(text,TextToSpeech.QUEUE_FLUSH,null,"confirmed-caption")==TextToSpeech.ERROR)
                speechStatus.setText("Speech could not start. Your confirmed caption is still visible.");
        });
        EditText reply=new EditText(this);reply.setHint("Type a reply for the signer");reply.setTextColor(Color.WHITE);reply.setHintTextColor(Color.LTGRAY);reply.setTextSize(24);root.addView(reply);
        benchmarkButton=new Button(this);benchmarkButton.setText("Run 10-minute device benchmark");root.addView(benchmarkButton);
        benchmarkButton.setOnClickListener(v->worker.execute(()-> {
            if(!ready||recording) return;
            if(benchmarking) finishBenchmark();
            else {
                sequence.clear();frameTimes.clear();classifierTimes.clear();
                while(samples.length()>0)samples.remove(samples.length()-1);
                benchmarking=true;benchmarkStart=SystemClock.elapsedRealtime();
                runOnUiThread(()->{stopSpeech();clearChoices();status.setText("Benchmark running. Keep a signer in view. Captions paused.");benchmarkButton.setText("Stop benchmark early");});
            }
        }));
        Button export=new Button(this);export.setText("Export benchmark evidence");root.addView(export);
        export.setOnClickListener(v->{Intent intent=new Intent(Intent.ACTION_CREATE_DOCUMENT);intent.setType("application/json");intent.addCategory(Intent.CATEGORY_OPENABLE);intent.putExtra(Intent.EXTRA_TITLE,"echogesture-device-evidence.json");startActivityForResult(intent,42);});
        speech=new TextToSpeech(this,result->runOnUiThread(()->configureOfflineSpeech(result)));
        worker.execute(()-> {
            try {
                recognizer=new SignEngine(this);
                holistic=HolisticLandmarker.createFromOptions(this,HolisticLandmarker.HolisticLandmarkerOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setModelAssetPath("holistic_landmarker.task").setDelegate(com.google.mediapipe.tasks.core.Delegate.CPU).build())
                    .setRunningMode(RunningMode.VIDEO).build());
                ready=true;
                runOnUiThread(()->status.setText("Ready. Use the rear camera to face the signer."));
            } catch(Exception e){showError(e);}
        });
        if(ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)==PackageManager.PERMISSION_GRANTED) startCamera();
        else requestPermissions(new String[]{Manifest.permission.CAMERA},7);
    }
    private void clearChoices(){pending=null;for(Button button:choices)button.setVisibility(View.GONE);}
    private void refreshCaption(){
        String text=confirmed.isEmpty()?"Your confirmed words appear here.":String.join(" ",confirmed).replace('_',' ').toLowerCase(Locale.US);
        caption.setText(text);if(captionHud!=null)captionHud.setText(confirmed.isEmpty()?"Confirmed captions will appear here.":text);
        if(speakButton!=null)speakButton.setEnabled(speechReady&&!confirmed.isEmpty());
    }
    private void stopSpeech(){if(speech!=null)speech.stop();}
    private void configureOfflineSpeech(int result){
        if(destroyed||speech==null)return;
        if(result!=TextToSpeech.SUCCESS){speechStatus.setText("Offline speech is unavailable. You can still use captions.");return;}
        Voice selected=null;
        Set<Voice> voices=speech.getVoices();
        if(voices!=null)for(Voice voice:voices){
            if(voice.isNetworkConnectionRequired()||!"en".equals(voice.getLocale().getLanguage())||
                (voice.getFeatures()!=null&&voice.getFeatures().contains(TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED)))continue;
            if(selected==null||voice.getQuality()>selected.getQuality())selected=voice;
        }
        if(selected==null||speech.setVoice(selected)!=TextToSpeech.SUCCESS){
            speechStatus.setText("Install an offline English voice in your phone's speech settings to speak captions.");return;
        }
        speech.setOnUtteranceProgressListener(new UtteranceProgressListener(){
            @Override public void onStart(String id){}
            @Override public void onDone(String id){}
            @Override public void onError(String id){runOnUiThread(()->{if(!destroyed)speechStatus.setText("Speech could not play. Your confirmed caption is still visible.");});}
        });
        speechReady=true;speechStatus.setText("Speech uses an installed offline English voice.");refreshCaption();
    }
    @Override public void onRequestPermissionsResult(int code,String[] permissions,int[] grants){super.onRequestPermissionsResult(code,permissions,grants);if(code==7&&grants.length>0&&grants[0]==PackageManager.PERMISSION_GRANTED)startCamera();}
    private void startCamera(){
        ListenableFuture<ProcessCameraProvider> future=ProcessCameraProvider.getInstance(this);
        future.addListener(()-> {try {
            cameraProvider=future.get();
            Preview cameraPreview=new Preview.Builder().build();cameraPreview.setSurfaceProvider(preview.getSurfaceProvider());
            ImageAnalysis analysis=new ImageAnalysis.Builder().setTargetResolution(new android.util.Size(640,480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build();
            analysis.setAnalyzer(worker,this::analyze);
            cameraProvider.bindToLifecycle(this,CameraSelector.DEFAULT_BACK_CAMERA,cameraPreview,analysis);
        }catch(Exception e){showError(e);}},ContextCompat.getMainExecutor(this));
    }
    private static void copy(List<NormalizedLandmark> source,float[][] target,int offset){
        for(int i=0;i<source.size()&&i+offset<target.length;i++){target[i+offset][0]=source.get(i).x();target[i+offset][1]=source.get(i).y();target[i+offset][2]=source.get(i).z();}
    }
    private void analyze(ImageProxy image){
        Bitmap bitmap=null,rotated=null;
        try {
            if(destroyed||!ready||(!recording&&!benchmarking))return;
            long start=System.nanoTime();
            bitmap=image.toBitmap();
            Matrix rotation=new Matrix();rotation.postRotate(image.getImageInfo().getRotationDegrees());
            rotated=Bitmap.createBitmap(bitmap,0,0,bitmap.getWidth(),bitmap.getHeight(),rotation,true);
            long timestamp=Math.max(lastTimestamp+1,SystemClock.uptimeMillis());lastTimestamp=timestamp;
            HolisticLandmarkerResult result;
            try(MPImage mpImage=new BitmapImageBuilder(rotated).build()){result=holistic.detectForVideo(mpImage,timestamp);}
            double elapsed=(System.nanoTime()-start)/1e6;
            float[][] points=new float[75][3];copy(result.poseLandmarks(),points,0);copy(result.leftHandLandmarks(),points,33);copy(result.rightHandLandmarks(),points,54);
            handsSeen|=!result.leftHandLandmarks().isEmpty()||!result.rightHandLandmarks().isEmpty();
            sequence.add(points);
            if(benchmarking){
                while(sequence.size()>32)sequence.remove(0);
                if(sequence.size()==32){
                    SignEngine.Prediction prediction=recognizer.predict(sequence);
                    frameTimes.add(elapsed);classifierTimes.add(prediction.latencyMs);
                    JSONObject sample=new JSONObject();sample.put("elapsed_s",(SystemClock.elapsedRealtime()-benchmarkStart)/1000.0);sample.put("landmark_stage_ms",elapsed);sample.put("classifier_ms",prediction.latencyMs);
                    if(Build.VERSION.SDK_INT>=29)sample.put("thermal_status",((PowerManager)getSystemService(POWER_SERVICE)).getCurrentThermalStatus());
                    BatteryManager battery=(BatteryManager)getSystemService(BATTERY_SERVICE);sample.put("battery_current_now_uA",battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW));samples.put(sample);
                }
                if(SystemClock.elapsedRealtime()-benchmarkStart>=600000)finishBenchmark();
            }else if(sequence.size()>=240||SystemClock.elapsedRealtime()-recordingStart>=10000)finishSign();
        }catch(Exception e){recording=false;benchmarking=false;showError(e);}
        finally{if(rotated!=null&&rotated!=bitmap)rotated.recycle();if(bitmap!=null)bitmap.recycle();image.close();}
    }
    private void finishSign(){
        recording=false;
        try {
            if(sequence.size()<3||!handsSeen){runOnUiThread(()->{status.setText("Hands were not clear. Please try again.");recordButton.setText("Start one sign");});return;}
            SignEngine.Prediction prediction=recognizer.predict(sequence);
            runOnUiThread(()->{pending=prediction.words;status.setText(prediction.confidence<0.65?"I am unsure. Choose a word or try again.":"Please check the word before adding it.");
                timing.setText(String.format(Locale.US,"Geometry + classifier %.1f ms; camera stage separate.",prediction.latencyMs));
                for(int i=0;i<prediction.words.length;i++){choices.get(i).setText("Add "+prediction.words[i]);choices.get(i).setVisibility(View.VISIBLE);}recordButton.setText("Start one sign");});
        }catch(Exception e){showError(e);}finally{sequence.clear();}
    }
    private void finishBenchmark(){benchmarking=false;sequence.clear();runOnUiThread(()->{status.setText("Benchmark stopped. Export evidence for analysis.");benchmarkButton.setText("Run 10-minute device benchmark");});}
    @Override protected void onActivityResult(int request,int result,Intent data){super.onActivityResult(request,result,data);if(request==42&&result==RESULT_OK&&data!=null&&data.getData()!=null){android.net.Uri uri=data.getData();worker.execute(()->{try {
        JSONObject report=new JSONObject();report.put("device",Build.MANUFACTURER+" "+Build.MODEL);report.put("android",Build.VERSION.RELEASE);report.put("samples",samples);report.put("scope","Camera conversion + Holistic stage and geometry + classifier measured separately; frame delivery/HUD delay excluded. Battery current is device-wide, not app power.");
        try(java.io.OutputStream out=getContentResolver().openOutputStream(uri)){if(out==null)throw new java.io.IOException("Cannot write export");out.write(report.toString(2).getBytes(java.nio.charset.StandardCharsets.UTF_8));}
        runOnUiThread(()->status.setText("Evidence exported. Review it before claiming a pass."));
    }catch(Exception e){showError(e);}});}}
    private void showError(Exception e){runOnUiThread(()->{status.setText("Unable to continue: "+e.getClass().getSimpleName()+" "+e.getMessage());recordButton.setText("Start one sign");benchmarkButton.setText("Run 10-minute device benchmark");});}
    @Override protected void onPause(){stopSpeech();super.onPause();worker.execute(()->{recording=false;benchmarking=false;sequence.clear();});}
    @Override protected void onDestroy(){destroyed=true;if(speech!=null){speech.stop();speech.shutdown();speech=null;}if(cameraProvider!=null)cameraProvider.unbindAll();worker.execute(()->{try{if(holistic!=null)holistic.close();if(recognizer!=null)recognizer.close();}catch(Exception ignored){}});worker.shutdown();super.onDestroy();}
}
