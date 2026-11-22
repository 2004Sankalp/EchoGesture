package org.echogesture.app;

import android.content.Context;
import ai.onnxruntime.*;
import org.json.JSONArray;
import java.nio.FloatBuffer;
import java.util.*;

/** Frozen 2000-class INT8-QDQ ensemble; all execution stays on device. */
public final class SignEngine implements AutoCloseable {
    private final OrtEnvironment environment;
    private final ArrayList<OrtSession> sessions=new ArrayList<>();
    private final org.json.JSONObject config;
    private final String[] labels;
    public static final class Prediction {
        public final String[] words;
        public final float confidence;
        public final double latencyMs;
        Prediction(String[] w, float c, double l) { words=w; confidence=c; latencyMs=l; }
    }
    public SignEngine(Context context) throws Exception {
        environment = OrtEnvironment.getEnvironment();
        environment.setTelemetry(false);
        config=new org.json.JSONObject(new String(context.getAssets().open("ensemble/ensemble.json").readAllBytes(),java.nio.charset.StandardCharsets.UTF_8));
        JSONArray members=config.getJSONArray("members");
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setIntraOpNumThreads(1); options.setInterOpNumThreads(1);
            for(int i=0;i<members.length();i++) sessions.add(environment.createSession(context.getAssets().open("ensemble/"+members.getJSONObject(i).getString("model")).readAllBytes(),options));
        }
        JSONArray data = new JSONArray(new String(context.getAssets().open("labels.json").readAllBytes(), java.nio.charset.StandardCharsets.UTF_8));
        labels = new String[data.length()];
        for (int i=0;i<labels.length;i++) labels[i]=data.getString(i);
    }
    public Prediction predict(List<float[][]> sequence) throws Exception {
        if(sequence.isEmpty())throw new IllegalArgumentException("Empty sign");
        long start=System.nanoTime();
        float[] logits=new float[labels.length];
        JSONArray members=config.getJSONArray("members");
        HashMap<String,float[][]> cache=new HashMap<>();
        for(int i=0;i<members.length();i++){
            org.json.JSONObject member=members.getJSONObject(i);
            String version=member.getString("feature_version");int frames=member.getInt("frames");
            String key=version+":"+frames;
            if(!cache.containsKey(key))cache.put(key,Features3D.prepare(sequence,frames,version.endsWith("v3")));
            float[][] features=cache.get(key);JSONArray scales=member.getJSONArray("scales");
            for(int v=0;v<scales.length();v++){
                float factor=(float)scales.getDouble(v);float[][] view=factor==1?features:Features3D.stretch(features,factor);
                try(OnnxTensor tensor=OnnxTensor.createTensor(environment,FloatBuffer.wrap(Features3D.flatten(view)),new long[]{1,308,frames});
                    OrtSession.Result result=sessions.get(i).run(Collections.singletonMap("features3d",tensor))){
                    float[] output=((float[][])result.get(0).getValue())[0];float weight=(float)member.getDouble("weight")/scales.length();
                    for(int c=0;c<logits.length;c++)logits[c]+=weight*output[c];
                }
            }
        }
        double ms=(System.nanoTime()-start)/1e6;
        Integer[] order=new Integer[logits.length];
        for(int i=0;i<order.length;i++) order[i]=i;
        Arrays.sort(order,(a,b)->Float.compare(logits[b],logits[a]));
        double sum=0;
        for(float logit:logits) sum+=Math.exp(logit-logits[order[0]]);
        String[] words=new String[Math.min(5,order.length)];
        for(int i=0;i<words.length;i++) words[i]=labels[order[i]];
        return new Prediction(words,(float)(1/sum),ms);
    }
    public void close() throws OrtException { for(OrtSession session:sessions)session.close(); }
}
