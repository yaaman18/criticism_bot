#include "ofMain.h"
#include "ofApp.h"

int main() {
    ofGLWindowSettings settings;
    settings.setGLVersion(3, 2);
    settings.setSize(1440, 960);
    ofCreateWindow(settings);
    ofRunApp(std::make_shared<ofApp>());
}
