#pragma once

#include "ofMain.h"

struct FrameTextures {
    ofTexture life;
    ofTexture field;
    ofTexture body;
    ofTexture aura;
};

class ofApp : public ofBaseApp {
public:
    void setup() override;
    void update() override;
    void draw() override;
    void keyPressed(int key) override;

private:
    bool loadSession(const ofFilePath& manifestPath);
    bool loadFrame(std::size_t frameIndex);
    void updatePlayback();
    void drawHud() const;

    ofShader shader_;
    ofPlanePrimitive quad_;
    std::vector<ofJson> frames_;
    FrameTextures current_;

    std::size_t frameIndex_ = 0;
    float playhead_ = 0.0f;
    float playbackRate_ = 10.0f;
    bool playing_ = true;

    float overlayMix_ = 0.72f;
    float pulseStrength_ = 0.22f;
    float exposure_ = 1.15f;
    float chromaWarp_ = 0.018f;

    std::string sessionRoot_;
    std::string sessionManifest_;
};
