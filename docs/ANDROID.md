# Android handoff

The ARM64 debug APK is `releases/EchoGesture-debug.apk`, version **1.0-submission**, version code 3, minimum SDK 26 and target SDK 35. It built successfully. `aapt` confirms CAMERA and no INTERNET or ACCESS_NETWORK_STATE. Physical realme GT 6T / Android 16 operation is not yet validated.

## Quick check before submission

1. Install the APK on the realme and allow camera access. Use the rear camera facing the signer. If Android reports a signature conflict with an earlier prototype, uninstall that prototype before installing this debug build.
2. Tap **Start one sign**, make one isolated sign, then **Review this sign**. Review the five candidates and confirm a word. Try **Undo**, **Clear** and a typed reply.
3. **Speak confirmed words** needs an installed offline English voice. It only speaks confirmed captions and does not automatically speak predictions. If no offline voice is present, captions remain usable.
4. If there is enough time, use **Run 10-minute device benchmark** and **Export benchmark evidence**. Keep a signer in view. Stopping early is not a ten-minute thermal pass. Accuracy still requires labelled clips; paired robustness needs predetermined clean/perturbed videos. Whole-device battery readings do not by themselves establish average app power.

Active Android inference assets total **19.276143MB**, including the Holistic model, both classifiers, labels and config. APK size is **60.83MB** because it also contains native runtimes and application resources. Holistic is FLOAT16 and the QDQ classifier graph has float operators; do not call it a wholly integer pipeline. Phone latency, power, robustness and thermal metrics remain NOT_MEASURED until actual evidence is collected.

The selected models and configuration are frozen before the public-pose test. Do not tune the model from final-test outcomes. The public-pose result does not imply the same video or phone accuracy. The leader may add a real 2–3 minute recording if time permits. See `START_HERE_TEAM_LEADER.txt` and `docs/Current-Evidence.pdf`.

## Build from source

Use Java 17 or newer, Android SDK 35 and the supplied Gradle wrapper:

```sh
cd src/android
./gradlew assembleDebug
```

Windows: `gradlew.bat assembleDebug`. Set the local SDK path or ANDROID_HOME for your machine. Dependencies may need an initial network download. Gradle copies only the frozen ensemble, labels and Holistic into the APK. No private keystore is included; the supplied APK is a debug build, not a Play Store release.
