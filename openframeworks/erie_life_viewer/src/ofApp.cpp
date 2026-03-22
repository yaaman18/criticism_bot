#include "ofApp.h"

namespace {
std::string joinPath(const std::string& left, const std::string& right) {
    return ofFilePath::join(left, right);
}
}

void ofApp::setup() {
    ofDisableArbTex();
    ofSetFrameRate(60);
    ofBackground(4, 3, 7);
    quad_.set(ofGetWidth(), ofGetHeight(), 2, 2);
    shader_.load("shaders/life.vert", "shaders/life.frag");

    const auto defaultManifest = ofToDataPath("session/manifest.json", true);
    if (!loadSession(defaultManifest)) {
        ofLogWarning() << "No default session found at " << defaultManifest;
    }
}

void ofApp::update() {
    updatePlayback();
}

void ofApp::draw() {
    ofSetColor(255);
    if (!frames_.empty()) {
        shader_.begin();
        shader_.setUniformTexture("uLifeTex", current_.life, 1);
        shader_.setUniformTexture("uFieldTex", current_.field, 2);
        shader_.setUniformTexture("uBodyTex", current_.body, 3);
        shader_.setUniformTexture("uAuraTex", current_.aura, 4);
        shader_.setUniform2f("uResolution", ofGetWidth(), ofGetHeight());
        shader_.setUniform1f("uTime", ofGetElapsedTimef());
        shader_.setUniform1f("uOverlayMix", overlayMix_);
        shader_.setUniform1f("uPulseStrength", pulseStrength_);
        shader_.setUniform1f("uExposure", exposure_);
        shader_.setUniform1f("uChromaWarp", chromaWarp_);
        quad_.draw();
        shader_.end();
    }
    drawHud();
}

void ofApp::keyPressed(int key) {
    if (key == ' ') {
        playing_ = !playing_;
    } else if (key == OF_KEY_RIGHT && !frames_.empty()) {
        frameIndex_ = (frameIndex_ + 1) % frames_.size();
        playhead_ = static_cast<float>(frameIndex_);
        loadFrame(frameIndex_);
    } else if (key == OF_KEY_LEFT && !frames_.empty()) {
        frameIndex_ = (frameIndex_ + frames_.size() - 1) % frames_.size();
        playhead_ = static_cast<float>(frameIndex_);
        loadFrame(frameIndex_);
    } else if (key == '[') {
        playbackRate_ = std::max(1.0f, playbackRate_ - 1.0f);
    } else if (key == ']') {
        playbackRate_ = std::min(60.0f, playbackRate_ + 1.0f);
    } else if (key == '1') {
        overlayMix_ = std::max(0.0f, overlayMix_ - 0.05f);
    } else if (key == '2') {
        overlayMix_ = std::min(1.0f, overlayMix_ + 0.05f);
    } else if (key == '3') {
        pulseStrength_ = std::max(0.0f, pulseStrength_ - 0.03f);
    } else if (key == '4') {
        pulseStrength_ = std::min(0.8f, pulseStrength_ + 0.03f);
    } else if (key == 'f') {
        ofToggleFullscreen();
    } else if (key == 'r' && !sessionManifest_.empty()) {
        loadSession(sessionManifest_);
    }
}

bool ofApp::loadSession(const ofFilePath& manifestPath) {
    ofFile file(manifestPath);
    if (!file.exists()) {
        return false;
    }
    const auto manifest = ofLoadJson(file.getAbsolutePath());
    if (!manifest.contains("frames") || manifest["frames"].empty()) {
        return false;
    }

    frames_.clear();
    for (const auto& frame : manifest["frames"]) {
        frames_.push_back(frame);
    }

    sessionManifest_ = file.getAbsolutePath();
    sessionRoot_ = file.getEnclosingDirectory();
    frameIndex_ = 0;
    playhead_ = 0.0f;
    return loadFrame(frameIndex_);
}

bool ofApp::loadFrame(std::size_t frameIndex) {
    if (frameIndex >= frames_.size()) {
        return false;
    }
    const auto& frame = frames_[frameIndex];

    ofImage life;
    ofImage field;
    ofImage body;
    ofImage aura;

    const bool ok =
        life.load(joinPath(sessionRoot_, frame["life"].get<std::string>())) &&
        field.load(joinPath(sessionRoot_, frame["field"].get<std::string>())) &&
        body.load(joinPath(sessionRoot_, frame["body"].get<std::string>())) &&
        aura.load(joinPath(sessionRoot_, frame["aura"].get<std::string>()));

    if (!ok) {
        return false;
    }

    current_.life = life.getTexture();
    current_.field = field.getTexture();
    current_.body = body.getTexture();
    current_.aura = aura.getTexture();
    return true;
}

void ofApp::updatePlayback() {
    if (!playing_ || frames_.empty()) {
        return;
    }
    playhead_ += ofGetLastFrameTime() * playbackRate_;
    const auto nextIndex = static_cast<std::size_t>(playhead_) % frames_.size();
    if (nextIndex != frameIndex_) {
        frameIndex_ = nextIndex;
        loadFrame(frameIndex_);
    }
}

void ofApp::drawHud() const {
    ofPushStyle();
    ofSetColor(240);
    ofDrawBitmapStringHighlight(
        "ERIE Life Viewer\n"
        "space play/pause\n"
        "left/right scrub\n"
        "[ ] speed\n"
        "1/2 overlay\n"
        "3/4 pulse\n"
        "r reload\n"
        "frame: " + ofToString(frameIndex_) + "/" + ofToString(std::max<std::size_t>(1, frames_.size()) - 1) + "\n"
        "fps: " + ofToString(ofGetFrameRate(), 1),
        24,
        28,
        ofColor(10, 10, 14, 180),
        ofColor(245, 245, 250)
    );
    ofPopStyle();
}
