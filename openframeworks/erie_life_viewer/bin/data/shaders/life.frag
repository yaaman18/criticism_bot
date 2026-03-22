#version 150

uniform sampler2DRect uLifeTex;
uniform sampler2DRect uFieldTex;
uniform sampler2DRect uBodyTex;
uniform sampler2DRect uAuraTex;
uniform vec2 uResolution;
uniform float uTime;
uniform float uOverlayMix;
uniform float uPulseStrength;
uniform float uExposure;
uniform float uChromaWarp;

in vec2 vTexCoord;
out vec4 outputColor;

vec3 tonemap(vec3 x) {
    return x / (1.0 + x);
}

void main() {
    vec2 uv = vTexCoord;
    vec2 center = uv / uResolution;

    vec3 body = texture(uBodyTex, uv).rgb;
    vec3 aura = texture(uAuraTex, uv).rgb;
    vec3 field = texture(uFieldTex, uv).rgb;

    float bodyPulse = sin(uTime * 2.3 + body.b * 6.2831) * 0.5 + 0.5;
    float fieldPulse = sin(uTime * 0.8 + field.g * 4.0) * 0.5 + 0.5;

    vec2 warp =
        (field.rg - 0.5) * (8.0 * uChromaWarp) +
        (aura.rb - 0.5) * (14.0 * uChromaWarp) +
        (body.gb - 0.5) * (4.0 * uChromaWarp);

    vec3 lifeR = texture(uLifeTex, uv + warp * 1.30).rgb;
    vec3 lifeG = texture(uLifeTex, uv + warp * 0.75).rgb;
    vec3 lifeB = texture(uLifeTex, uv - warp * 1.10).rgb;
    vec3 life = vec3(lifeR.r, lifeG.g, lifeB.b);

    float edgeGlow = smoothstep(0.32, 0.98, body.b);
    float membraneGlow = smoothstep(0.18, 0.95, body.r) * (0.35 + 0.65 * bodyPulse);
    float uncertaintyMist = smoothstep(0.08, 0.95, aura.b) * (0.25 + 0.75 * fieldPulse);

    vec3 fieldTint = vec3(
        0.95 * field.r + 0.15 * field.g,
        0.65 * field.g + 0.20 * field.b,
        0.75 * field.b + 0.10 * field.r
    );

    vec3 core = life * vec3(1.18, 1.03, 1.32);
    vec3 bodyTint = mix(vec3(0.24, 0.85, 1.00), vec3(1.00, 0.52, 0.86), body.g);
    vec3 auraTint = mix(vec3(0.08, 0.12, 0.18), vec3(0.56, 0.24, 0.88), aura.r);

    vec3 color = core;
    color += fieldTint * uOverlayMix * 0.45;
    color += bodyTint * edgeGlow * 0.50;
    color += vec3(0.85, 0.92, 1.00) * membraneGlow * (0.45 + 0.55 * uPulseStrength);
    color += auraTint * uncertaintyMist * 0.60;

    float vignette = smoothstep(1.15, 0.18, distance(center, vec2(0.5)));
    color *= (0.65 + 0.35 * vignette);
    color *= uExposure;
    color = tonemap(color);
    color = pow(color, vec3(0.92));
    outputColor = vec4(color, 1.0);
}
