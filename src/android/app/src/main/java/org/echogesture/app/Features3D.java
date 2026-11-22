package org.echogesture.app;

import java.util.*;

/** Java port of palm3d-context-v2/v3; inferred depth is not calibrated depth. */
public final class Features3D {
    static final int[] BODY={0,11,12,13,14,15,16};
    static float norm(float[] a){return (float)Math.sqrt(dot(a,a));}
    static float dot(float[] a,float[] b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
    static float[] sub(float[] a,float[] b){return new float[]{a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
    static float[] cross(float[] a,float[] b){return new float[]{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
    static boolean present(float[] a){return a[0]!=0||a[1]!=0||a[2]!=0;}
    static float clip(float v){return Math.max(-8,Math.min(8,v));}
    static float[] rotate(float[] v,float ex,float ey){return new float[]{ex*v[0]+ey*v[1],-ey*v[0]+ex*v[1],v[2]};}
    static final class Hand {
        float[][] shape=new float[21][3],basis=new float[3][3];boolean valid;
        Hand(float[][] p){
            float[][] local=new float[21][3];for(int j=0;j<21;j++)local[j]=sub(p[j],p[0]);
            float[] lateral=sub(local[5],local[17]),forward=local[9].clone();
            float lat=norm(lateral),scale=(lat+norm(forward))/2;
            float[] x=lateral.clone();for(int c=0;c<3;c++)x[c]/=Math.max(lat,1e-7f);
            float projection=dot(forward,x);for(int c=0;c<3;c++)forward[c]-=projection*x[c];
            float length=norm(forward);valid=scale>1e-6&&lat>1e-6&&length>.05f*scale;
            if(!valid)return;
            float[] y=forward.clone();for(int c=0;c<3;c++)y[c]/=Math.max(length,1e-7f);
            float[] z=cross(x,y);float[][] axes={x,y,z};
            for(int r=0;r<3;r++)for(int c=0;c<3;c++)basis[r][c]=axes[c][r];
            for(int j=0;j<21;j++)for(int c=0;c<3;c++)shape[j][c]=dot(local[j],axes[c])/Math.max(scale,1e-7f);
        }
    }
    public static float[][] prepare(List<float[][]> sequence,int frames,boolean bodyFrame){
        if(sequence.isEmpty())throw new IllegalArgumentException("Empty sign");
        ArrayList<Float> lengths=new ArrayList<>();
        for(float[][] p:sequence){float d=norm(sub(p[11],p[12]));if(present(p[11])&&present(p[12])&&d>1e-5)lengths.add(d);}
        Collections.sort(lengths);int n=lengths.size();float scale=n==0?1:(lengths.get((n-1)/2)+lengths.get(n/2))/2;
        int count=sequence.size();float[][] described=new float[count][308];float[][] previous=null;boolean[] previousPresent=null;boolean previousShoulders=false;
        for(int t=0;t<count;t++){
            float[][] p=sequence.get(t);float[] row=described[t];boolean shoulders=present(p[11])&&present(p[12]);
            float[] centre=new float[3];for(int c=0;c<3;c++)centre[c]=(p[11][c]+p[12][c])/2;
            float ex=1,ey=0;float dx=p[11][0]-p[12][0],dy=p[11][1]-p[12][1];float len=(float)Math.hypot(dx,dy);
            if(bodyFrame&&shoulders&&len>1e-5){ex=dx/len;ey=dy/len;}
            float[][] body=new float[7][3];boolean[] seen=new boolean[7];
            for(int j=0;j<7;j++){
                seen[j]=present(p[BODY[j]]);row[21+j]=seen[j]?1:0;
                if(seen[j]&&shoulders){float[] v=sub(p[BODY[j]],centre);for(int c=0;c<3;c++)v[c]/=scale;body[j]=rotate(v,ex,ey);}
                for(int c=0;c<3;c++)row[j*3+c]=body[j][c];
            }
            for(int side=0;side<2;side++){
                int start=33+side*21,shape=side==0?28:101,basis=side==0?91:164,valid=side==0?100:173,raw=side==0?180:244;
                float[][] hand=Arrays.copyOfRange(p,start,start+21);Hand h=new Hand(hand);row[valid]=h.valid?1:0;
                for(int j=0;j<21;j++)for(int c=0;c<3;c++)row[shape+3*j+c]=h.shape[j][c];
                for(int c=0;c<3;c++){float[] column={h.basis[0][c],h.basis[1][c],h.basis[2][c]};column=rotate(column,ex,ey);for(int r=0;r<3;r++)row[basis+3*r+c]=column[r];}
                boolean handPresent=false;for(float[] point:hand)handPresent|=present(point);
                if(handPresent&&shoulders){
                    for(int j=0;j<21;j++){float[] v={(hand[j][0]-centre[0])/scale,(hand[j][1]-centre[1])/scale,(hand[j][2]-hand[0][2])/scale};v=rotate(v,ex,ey);for(int c=0;c<3;c++)row[raw+3*j+c]=v[c];}
                    row[raw+63]=norm(sub(hand[9],hand[0]))/scale;
                }
                if(t>0&&shoulders&&previousShoulders&&seen[5+side]&&previousPresent[5+side])for(int c=0;c<3;c++)row[174+3*side+c]=body[5+side][c]-previous[5+side][c];
            }
            previous=body;previousPresent=seen;previousShoulders=shoulders;
            for(int c=0;c<308;c++)row[c]=clip(row[c]);
        }
        float[][] output=new float[308][frames];
        for(int f=0;f<frames;f++){int index=(int)Math.rint((double)f*(count-1)/(frames-1));for(int c=0;c<308;c++)output[c][f]=described[index][c];}
        return output;
    }
    public static float[][] stretch(float[][] input,float factor){
        int frames=input[0].length;float[][] x=new float[308][];for(int c=0;c<308;c++)x[c]=input[c].clone();
        for(int j=0;j<7;j++)for(int f=0;f<frames;f++)x[3*j+1][f]*=factor;
        for(int s=0;s<2;s++)for(int f=0;f<frames;f++)x[174+3*s+1][f]*=factor;
        for(int side=0;side<2;side++){
            int raw=side==0?180:244,shape=side==0?28:101,basis=side==0?91:164,valid=side==0?100:173;
            for(int f=0;f<frames;f++){
                float[][] hand=new float[21][3];boolean any=false;
                for(int j=0;j<21;j++)for(int c=0;c<3;c++){hand[j][c]=x[raw+3*j+c][f]*(c==1?factor:1);any|=hand[j][c]!=0;x[raw+3*j+c][f]=hand[j][c];}
                Hand h=new Hand(hand);boolean observed=any&&x[valid][f]>0;
                x[raw+63][f]=norm(sub(hand[9],hand[0]));
                if(observed){for(int j=0;j<21;j++)for(int c=0;c<3;c++)x[shape+3*j+c][f]=h.shape[j][c];for(int r=0;r<3;r++)for(int c=0;c<3;c++)x[basis+3*r+c][f]=h.basis[r][c];x[valid][f]=h.valid?1:0;}
            }
        }
        for(float[] channel:x)for(int f=0;f<frames;f++)channel[f]=clip(channel[f]);return x;
    }
    public static float[] flatten(float[][] features){int frames=features[0].length;float[] values=new float[features.length*frames];for(int c=0;c<features.length;c++)System.arraycopy(features[c],0,values,c*frames,frames);return values;}
}
