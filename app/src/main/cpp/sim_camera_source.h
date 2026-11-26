#pragma once
// Simulated MG4 surround cameras for the Android (Automotive) emulator.
//
// Only compiled into the "sim" build type (MG4_CAMERA_SIM). It replaces the four V4L2 nodes
// /dev/video14..17 with a ray-traced parking scene seen through four fisheye cameras whose
// parameters are exactly those of SimulatedCameras.java (the calibration the Java side uses in
// the same build), so the bird's-eye, 3D bowl, wheel and side views all line up.
//
// Output contract = the real cameras: 720x480 UYVY, two fields stacked top-first
// (rows 0..239 = even lines, rows 240..479 = odd lines), V4L2_FIELD_NONE, unmirrored.
//
// World frame (same as the vehicle calibration): millimetres, x to the right of the car,
// y towards the rear, z DOWN, ground at z = 0, origin at the centre of the rear axle.
//
// Self-contained on purpose (no OpenCV, no Android) so it can be tested on a desktop.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace mg4sim {

static constexpr int W = 720, H = 480, FIELD_H = 240, STRIDE = W * 2;
static constexpr size_t FRAME_BYTES = static_cast<size_t>(STRIDE) * H;
static constexpr int FPS = 25;

struct V3 { double x, y, z; };
static inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
static inline double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static inline V3 norm(V3 a) { double l = std::sqrt(dot(a, a)); return l > 0 ? a * (1.0 / l) : a; }

// ------------------------------------------------------------------ camera rig
// KEEP IN SYNC with app/src/main/java/com/drivehub/kamera/birdseye/SimulatedCameras.java
// Order = calibration order: left, right, front, rear.
struct CamSpec {
    double fx, fy, cx, cy, k[4];
    V3 center, forward, right;   // world; forward/right are unit and orthogonal
};

static inline const CamSpec *camSpecs() {
    static const double cs = std::cos(65 * M_PI / 180), ss = std::sin(65 * M_PI / 180);
    static const double cf = std::cos(40 * M_PI / 180), sf = std::sin(40 * M_PI / 180);
    static const double cr = std::cos(45 * M_PI / 180), sr = std::sin(45 * M_PI / 180);
    static const CamSpec specs[4] = {
        // left mirror, looking left and 65 deg down, image right = car front
        {230, 230, 360, 240, {0.015, -0.004, 0.0005, 0}, {-1040, -2250, -1050}, {-cs, 0, ss}, {0, -1, 0}},
        // right mirror, looking right and 65 deg down, image right = car rear
        {230, 230, 360, 240, {0.015, -0.004, 0.0005, 0}, {1040, -2250, -1050}, {cs, 0, ss}, {0, 1, 0}},
        // front grille, looking forward and 40 deg down, image right = car right
        {230, 230, 360, 240, {0.015, -0.004, 0.0005, 0}, {0, -3570, -620}, {0, -cf, sf}, {1, 0, 0}},
        // rear tailgate, looking back and 45 deg down, image right = car left (unmirrored)
        {230, 230, 360, 240, {0.015, -0.004, 0.0005, 0}, {0, 760, -980}, {0, cr, sr}, {-1, 0, 0}},
    };
    return specs;
}

/** /dev/videoN -> calibration slot (left, right, front, rear), or -1. */
static inline int slotForVideoIndex(int videoIndex) {
    switch (videoIndex) {
        case 16: return 0; // left
        case 14: return 1; // right
        case 15: return 2; // front
        case 17: return 3; // rear
        default: return -1;
    }
}

static inline double distortRadius(const CamSpec &c, double th) {
    double a = th * th;
    return th * (1 + a * (c.k[0] + a * (c.k[1] + a * (c.k[2] + a * c.k[3]))));
}

/** World point -> pixel (reference 720x480). Returns false behind the lens. */
static inline bool projectWorld(const CamSpec &c, V3 p, double &u, double &v) {
    V3 d = cross(c.forward, c.right);
    V3 q = p - c.center;
    double x = dot(q, c.right), y = dot(q, d), z = dot(q, c.forward);
    double r = std::sqrt(x * x + y * y), th = std::atan2(r, z);
    double s = r > 1e-12 ? distortRadius(c, th) / r : 0;
    u = c.cx + c.fx * x * s;
    v = c.cy + c.fy * y * s;
    return th < 1.75;
}

/** Pixel -> unit world ray. Returns false outside the lens circle. */
static inline bool pixelRay(const CamSpec &c, double u, double v, V3 &out, double &theta) {
    double nx = (u - c.cx) / c.fx, ny = (v - c.cy) / c.fy, rd = std::hypot(nx, ny);
    const double maxTh = 1.72; // ~98.5 deg: the edge of the simulated lens circle
    if (rd >= distortRadius(c, maxTh)) return false;
    double lo = 0, hi = maxTh;
    for (int i = 0; i < 40; i++) { double m = (lo + hi) / 2; (distortRadius(c, m) < rd ? lo : hi) = m; }
    theta = (lo + hi) / 2;
    double s = rd > 1e-12 ? std::sin(theta) / rd : 0;
    double lx = nx * s, ly = ny * s, lz = std::cos(theta);
    V3 d = cross(c.forward, c.right);
    out = norm(c.right * lx + d * ly + c.forward * lz);
    return true;
}

// ------------------------------------------------------------------ scene
struct RGB { double r, g, b; };
static inline RGB mix(RGB a, RGB b, double t) { return {a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t}; }
static inline RGB scale(RGB a, double s) { return {a.r * s, a.g * s, a.b * s}; }

static inline double hash2(int x, int y) {
    uint32_t h = static_cast<uint32_t>(x) * 374761393u + static_cast<uint32_t>(y) * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return ((h ^ (h >> 16)) & 0xffff) / 65535.0;
}
static inline double valueNoise(double x, double y) {
    int ix = static_cast<int>(std::floor(x)), iy = static_cast<int>(std::floor(y));
    double fx = x - ix, fy = y - iy;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
    double a = hash2(ix, iy), b = hash2(ix + 1, iy), c = hash2(ix, iy + 1), d = hash2(ix + 1, iy + 1);
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}

enum Mat { M_SKY, M_GROUND, M_SIDEWALK, M_WALL, M_OWN_CAR, M_TIRE, M_OTHER_CAR, M_PILLAR, M_POST };

struct Box { V3 lo, hi; Mat m; };

static inline const std::vector<Box> &boxes() {
    static const std::vector<Box> b = {
        // own car (MG4: 1836 x 4287, front overhang 841, rear overhang 741, wheelbase 2705)
        {{-918, -3546, -950}, {918, 741, -330}, M_OWN_CAR},      // lower body
        {{-770, -2650, -1504}, {770, 380, -950}, M_OWN_CAR},     // cabin
        {{-925, -3035, -660}, {-703, -2375, 0}, M_TIRE}, {{703, -3035, -660}, {925, -2375, 0}, M_TIRE},
        {{-925, -330, -660}, {-703, 330, 0}, M_TIRE},    {{703, -330, -660}, {925, 330, 0}, M_TIRE},
        // neighbour car parked on the left
        {{-3950, -3600, -1450}, {-2150, 650, -320}, M_OTHER_CAR},
        {{-3960, -3150, -640}, {-3760, -2500, 0}, M_TIRE}, {{-2340, -3150, -640}, {-2140, -2500, 0}, M_TIRE},
        {{-3960, -250, -640}, {-3760, 400, 0}, M_TIRE},    {{-2340, -250, -640}, {-2140, 400, 0}, M_TIRE},
        // concrete pillar front-right
        {{2500, -2000, -7000}, {2900, -1600, 0}, M_PILLAR},
        // raised sidewalk in front of the parking bays (curb 130 mm)
        {{-14000, -16000, -130}, {14000, -4650, 0}, M_SIDEWALK},
    };
    return b;
}

static inline bool hitBox(const Box &b, V3 o, V3 d, double &tHit, int &axis) {
    double t0 = 1e-3, t1 = 1e30; int ax = -1;
    const double os[3] = {o.x, o.y, o.z}, ds[3] = {d.x, d.y, d.z};
    const double lo[3] = {b.lo.x, b.lo.y, b.lo.z}, hi[3] = {b.hi.x, b.hi.y, b.hi.z};
    for (int i = 0; i < 3; i++) {
        if (std::fabs(ds[i]) < 1e-12) { if (os[i] < lo[i] || os[i] > hi[i]) return false; continue; }
        double a = (lo[i] - os[i]) / ds[i], c = (hi[i] - os[i]) / ds[i];
        if (a > c) std::swap(a, c);
        if (a > t0) { t0 = a; ax = i; }
        if (c < t1) t1 = c;
        if (t0 > t1) return false;
    }
    if (ax < 0) return false; // starts inside
    tHit = t0; axis = ax;
    return true;
}

static const V3 SUN = norm({0.45, 0.35, -0.82});

static inline double carShadow(double x, double y, double x0, double x1, double y0, double y1) {
    double dx = std::max({x0 - x, 0.0, x - x1}), dy = std::max({y0 - y, 0.0, y - y1});
    double d = std::hypot(dx, dy);
    return 1.0 - 0.55 * std::max(0.0, 1.0 - d / 450.0);
}

static inline RGB groundColor(double x, double y) {
    double n = valueNoise(x / 140, y / 140) * 0.35 + valueNoise(x / 18, y / 18) * 0.65;
    RGB c = {88 + 18 * n, 89 + 18 * n, 93 + 17 * n};
    // oil stains in the middle of each bay
    double stain = valueNoise(x / 350 + 7, y / 350 + 3);
    if (stain > 0.8 && std::fmod(std::fabs(x) + 1350, 2700) > 700 && std::fmod(std::fabs(x) + 1350, 2700) < 2000)
        c = scale(c, 0.9);
    const RGB white = {228, 228, 222}, yellow = {222, 182, 40};
    // parking bay lines (bays 2700 wide), from the curb back to y = 1200
    if (y > -4650 && y < 1200) {
        for (double lx : {-4050.0, -1350.0, 1350.0, 4050.0})
            if (std::fabs(x - lx) < 60) c = white;
    }
    if (y > 1140 && y < 1260 && std::fabs(x) < 6750) c = white;        // bay end line
    // driving lane: dashed centre line
    if (std::fabs(y - 4300) < 60 && std::fmod(std::fabs(x) + 10000, 2000) < 1100) c = white;
    // zebra crossing behind the car
    if (y > 6200 && y < 8400 && std::fabs(x) < 3000 && std::fmod(x + 10000, 900) < 450) c = yellow;
    // painted arrow in the lane, pointing left
    {
        double ax = x + 2200, ay = y - 2900;
        if (std::fabs(ay) < 110 && ax > -300 && ax < 900) c = white;
        if (ax <= -300 && ax > -900 && std::fabs(ay) < (ax + 900) * 0.55) c = white;
    }
    c = scale(c, carShadow(x, y, -918 - 60, 918 + 120, -3546, 741 + 80));
    c = scale(c, carShadow(x, y, -3950, -2150 + 120, -3600, 650 + 80));
    return c;
}

static inline RGB sidewalkTop(double x, double y) {
    // 400 mm paving slabs
    double fx = std::fmod(x + 40000, 400), fy = std::fmod(y + 40000, 400);
    double n = valueNoise(x / 60, y / 60);
    RGB c = {162 + 18 * n, 158 + 18 * n, 150 + 16 * n};
    if (fx < 18 || fy < 18) c = scale(c, 0.72);
    return c;
}

static inline RGB skyColor(V3 d) {
    double up = std::max(0.0, -d.z);
    return mix(RGB{205, 215, 226}, RGB{88, 136, 205}, std::pow(up, 0.6));
}

/** Static scene shading. dist receives the hit distance (1e9 for sky). */
static inline RGB trace(V3 o, V3 d, double &dist) {
    double best = 1e30; int bestBox = -1, bestAxis = 0;
    const auto &bx = boxes();
    for (size_t i = 0; i < bx.size(); i++) {
        double t; int ax;
        if (hitBox(bx[i], o, d, t, ax) && t < best) { best = t; bestBox = static_cast<int>(i); bestAxis = ax; }
    }
    double tg = d.z > 1e-9 ? -o.z / d.z : 1e30;
    // surrounding garage walls 13.5 m away, 7 m high
    double tw = 1e30; int wallAxis = -1;
    {
        const double lo[3] = {-13500, -16000, -7000}, hi[3] = {13500, 13500, 0};
        const double os[3] = {o.x, o.y, o.z}, ds[3] = {d.x, d.y, d.z};
        for (int i = 0; i < 3; i++) {
            if (std::fabs(ds[i]) < 1e-12) continue;
            double t = ((ds[i] > 0 ? hi[i] : lo[i]) - os[i]) / ds[i];
            if (t > 0 && t < tw) { tw = t; wallAxis = i; }
        }
    }
    if (bestBox >= 0 && best <= tg && best <= tw) {
        const Box &b = bx[bestBox];
        V3 p = o + d * best;
        V3 n = {0, 0, 0};
        double sgn = (bestAxis == 0 ? d.x : bestAxis == 1 ? d.y : d.z) > 0 ? -1 : 1;
        if (bestAxis == 0) n.x = sgn; else if (bestAxis == 1) n.y = sgn; else n.z = sgn;
        double lam = 0.55 + 0.45 * std::max(0.0, dot(n, SUN));
        RGB c{128, 128, 128};
        switch (b.m) {
            case M_OWN_CAR: {
                c = {56, 92, 158};                                   // MG blue
                if (p.z < -1020 && p.z > -1440 && bestAxis != 2) c = {24, 28, 36}; // glass
                if (p.z > -500) c = {36, 38, 42};                    // black sills
                if ((p.y < -3350 || p.y > 560) && p.z > -760) c = {40, 42, 46}; // bumpers
                double spec = std::pow(std::max(0.0, dot(n, SUN)), 8) * 60;
                c = {c.r + spec, c.g + spec, c.b + spec};
                break;
            }
            case M_TIRE: c = {26, 26, 28}; if (std::fabs(n.x) > 0.5 && p.z > -560 && p.z < -100) c = {150, 152, 158}; break;
            case M_OTHER_CAR: c = {150, 28, 34}; if (p.z < -1050) c = {28, 30, 38}; if (p.z > -480) c = {40, 40, 44}; break;
            case M_PILLAR: {
                double nn = valueNoise(p.x / 70 + p.y / 70, p.z / 70);
                c = {168 + 20 * nn, 166 + 20 * nn, 160 + 18 * nn};
                if (p.z > -900 && std::fmod(-p.z + (p.x + p.y) * 0.5 + 20000, 300) < 150) c = {225, 180, 30};
                if (p.z > -900 && std::fmod(-p.z + (p.x + p.y) * 0.5 + 20000, 300) >= 150) c = {30, 30, 30};
                break;
            }
            case M_SIDEWALK: c = bestAxis == 2 ? sidewalkTop(p.x, p.y) : RGB{190, 188, 182}; break;
            default: break;
        }
        dist = best;
        return scale(c, b.m == M_SIDEWALK && bestAxis == 2 ? 1.0 : lam);
    }
    if (tg <= tw) {
        V3 p = o + d * tg;
        dist = tg;
        return groundColor(p.x, p.y);
    }
    if (tw < 1e29 && wallAxis != 2) {
        V3 p = o + d * tw;
        dist = tw;
        double h = -p.z, a = wallAxis == 0 ? p.y : p.x;
        double nn = valueNoise(a / 120, h / 120);
        RGB c = {150 + 18 * nn, 146 + 18 * nn, 140 + 16 * nn};
        if (h < 1200) c = {110, 112, 118};                                // painted base
        if (std::fabs(h - 1200) < 60) c = {210, 170, 40};                 // yellow band
        if (h > 2600 && h < 3800 && std::fmod(a + 50000, 3000) < 1800) c = {60, 72, 88}; // windows
        if (h > 6400) c = {200, 200, 196};                                // ceiling edge lights
        dist = tw;
        return scale(c, wallAxis == 0 ? 0.9 : 1.0);
    }
    dist = 1e9;
    return skyColor(d);
}

// ------------------------------------------------------------------ tiny 5x7 font
static inline const uint8_t *glyph(char ch) {
    static const uint8_t blank[7] = {0};
    struct G { char c; uint8_t rows[7]; };
    static const G font[] = {
        {'0', {14, 17, 19, 21, 25, 17, 14}}, {'1', {4, 12, 4, 4, 4, 4, 14}}, {'2', {14, 17, 1, 2, 4, 8, 31}},
        {'3', {31, 2, 4, 2, 1, 17, 14}},     {'4', {2, 6, 10, 18, 31, 2, 2}}, {'5', {31, 16, 30, 1, 1, 17, 14}},
        {'6', {6, 8, 16, 30, 17, 17, 14}},   {'7', {31, 1, 2, 4, 8, 8, 8}},   {'8', {14, 17, 17, 14, 17, 17, 14}},
        {'9', {14, 17, 17, 15, 1, 2, 12}},   {':', {0, 12, 12, 0, 12, 12, 0}}, {'#', {10, 10, 31, 10, 31, 10, 10}},
        {'-', {0, 0, 0, 31, 0, 0, 0}},       {'A', {14, 17, 17, 31, 17, 17, 17}}, {'C', {14, 17, 16, 16, 16, 17, 14}},
        {'D', {28, 18, 17, 17, 17, 18, 28}}, {'E', {31, 16, 16, 30, 16, 16, 31}}, {'G', {14, 17, 16, 23, 17, 17, 15}},
        {'H', {17, 17, 17, 31, 17, 17, 17}}, {'I', {14, 4, 4, 4, 4, 4, 14}},   {'M', {17, 27, 21, 21, 17, 17, 17}},
        {'N', {17, 17, 25, 21, 19, 17, 17}}, {'O', {14, 17, 17, 17, 17, 17, 14}}, {'R', {30, 17, 17, 30, 20, 18, 17}},
        {'S', {15, 16, 16, 14, 1, 1, 30}},   {'T', {31, 4, 4, 4, 4, 4, 4}},   {'U', {17, 17, 17, 17, 17, 17, 14}},
        {'V', {17, 17, 17, 17, 17, 10, 4}},
    };
    for (const G &g : font) if (g.c == ch) return g.rows;
    return blank;
}

// ------------------------------------------------------------------ per-camera source
static inline uint8_t clamp8(double v) { return static_cast<uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v + 0.5); }

struct Precomputed {
    std::vector<float> ray;     // 3 per pixel, full 720x480 reference grid
    std::vector<float> depth;   // static scene distance
    std::vector<uint8_t> rgb;   // static scene colour
    std::vector<uint8_t> uyvy;  // static frame, stacked fields
};

static inline void writeUyvyPair(uint8_t *dst, const uint8_t *a, const uint8_t *b) {
    auto Y = [](const uint8_t *p) { return 16 + (65.481 * p[0] + 128.553 * p[1] + 24.966 * p[2]) / 255; };
    auto U = [](const uint8_t *p) { return 128 + (-37.797 * p[0] - 74.203 * p[1] + 112 * p[2]) / 255; };
    auto V = [](const uint8_t *p) { return 128 + (112 * p[0] - 93.786 * p[1] - 18.214 * p[2]) / 255; };
    dst[0] = clamp8((U(a) + U(b)) / 2);
    dst[1] = clamp8(Y(a));
    dst[2] = clamp8((V(a) + V(b)) / 2);
    dst[3] = clamp8(Y(b));
}

/** Byte offset of reference row v (0..479) inside the stacked-fields buffer. */
static inline size_t rowOffset(int v) {
    return static_cast<size_t>((v & 1) ? FIELD_H + v / 2 : v / 2) * STRIDE;
}

static inline std::shared_ptr<const Precomputed> buildPrecomputed(int slot) {
    auto p = std::make_shared<Precomputed>();
    const CamSpec &c = camSpecs()[slot];
    const size_t n = static_cast<size_t>(W) * H;
    p->ray.assign(n * 3, 0.f); p->depth.assign(n, -1.f); p->rgb.assign(n * 3, 0); p->uyvy.assign(FRAME_BYTES, 0);
    for (int v = 0; v < H; v++) for (int u = 0; u < W; u++) {
        size_t i = static_cast<size_t>(v) * W + u;
        V3 d; double th;
        uint8_t *px = &p->rgb[i * 3];
        if (!pixelRay(c, u + 0.5, v + 0.5, d, th)) { px[0] = px[1] = px[2] = 6; continue; }
        // 2x2 supersampling: thin painted lines stay smooth
        double r = 0, g = 0, b = 0, dist = 0;
        for (int s = 0; s < 4; s++) {
            V3 ds; double ths, dd;
            if (!pixelRay(c, u + 0.25 + 0.5 * (s & 1), v + 0.25 + 0.5 * (s >> 1), ds, ths)) ds = d;
            RGB col = trace(c.center, ds, dd);
            r += col.r; g += col.g; b += col.b; if (s == 0) dist = dd;
        }
        double vign = 1.0 - 0.35 * std::pow(th / 1.72, 3);   // lens vignetting
        double edge = std::min(1.0, (1.72 - th) / 0.05);     // soft lens circle
        px[0] = clamp8(r / 4 * vign * edge); px[1] = clamp8(g / 4 * vign * edge); px[2] = clamp8(b / 4 * vign * edge);
        p->ray[i * 3] = static_cast<float>(d.x); p->ray[i * 3 + 1] = static_cast<float>(d.y); p->ray[i * 3 + 2] = static_cast<float>(d.z);
        p->depth[i] = static_cast<float>(dist);
    }
    for (int v = 0; v < H; v++) {
        uint8_t *row = p->uyvy.data() + rowOffset(v);
        const uint8_t *src = &p->rgb[static_cast<size_t>(v) * W * 3];
        for (int u = 0; u < W; u += 2) writeUyvyPair(row + u * 2, src + u * 3, src + u * 3 + 3);
    }
    return p;
}

// The static scene takes a second or two per camera (more in an unoptimised build), so it is
// built once, in the background, for all four cameras at the first open. Until a camera's
// scene is ready its feed shows a grey frame with the live label.
struct PrecomputeCache {
    std::mutex m;
    std::shared_ptr<const Precomputed> ready[4];
    bool started = false;
};
static inline PrecomputeCache &precomputeCache() { static PrecomputeCache c; return c; }

static inline void startPrecompute() {
    PrecomputeCache &c = precomputeCache();
    std::lock_guard<std::mutex> lock(c.m);
    if (c.started) return;
    c.started = true;
    for (int k = 0; k < 4; k++) {
        std::thread([k]() {
            auto p = buildPrecomputed(k);
            PrecomputeCache &cc = precomputeCache();
            std::lock_guard<std::mutex> l(cc.m);
            cc.ready[k] = p;
        }).detach();
    }
}

static inline std::shared_ptr<const Precomputed> precomputed(int slot) {
    PrecomputeCache &c = precomputeCache();
    std::lock_guard<std::mutex> lock(c.m);
    return c.ready[slot];
}

class Source {
public:
    explicit Source(int videoIndex) : slot_(slotForVideoIndex(videoIndex)) {
        if (slot_ >= 0) startPrecompute();
    }
    bool valid() const { return slot_ >= 0; }

    /** Renders the frame for time `seconds` into a 720x480 UYVY stacked-fields buffer. */
    void render(uint8_t *out, double seconds) {
        if (pre_ == nullptr) pre_ = precomputed(slot_);
        if (pre_ != nullptr) {
            std::memcpy(out, pre_->uyvy.data(), FRAME_BYTES);
            drawPost(out, seconds);
        } else {
            for (size_t i = 0; i < FRAME_BYTES; i += 2) { out[i] = 128; out[i + 1] = 70; }
        }
        drawLabel(out);
    }

private:
    int slot_;
    std::shared_ptr<const Precomputed> pre_;
    std::vector<uint8_t> scratch_;

    // An orange/white traffic post circling the car once every 24 s: visible in every camera
    // in turn, so the stitching, dashcam and blind-spot tiles can be checked with motion.
    static V3 postPosition(double s) {
        double a = s * 2 * M_PI / 24.0;
        return {2300 * std::cos(a), -1400 + 3700 * std::sin(a), 0};
    }

    void drawPost(uint8_t *out, double seconds) {
        const CamSpec &c = camSpecs()[slot_];
        const V3 base = postPosition(seconds);
        const double R = 170, HGT = 900;
        // screen bounding box from the cylinder silhouette
        double u0 = 1e9, v0 = 1e9, u1 = -1e9, v1 = -1e9; bool any = false;
        for (int k = 0; k < 16; k++) for (double z : {0.0, -HGT}) {
            double a = k * M_PI / 8, u, v;
            V3 q = {base.x + R * std::cos(a), base.y + R * std::sin(a), z};
            if (!projectWorld(c, q, u, v)) continue;
            any = true; u0 = std::min(u0, u); u1 = std::max(u1, u); v0 = std::min(v0, v); v1 = std::max(v1, v);
        }
        if (!any) return;
        int iu0 = std::max(0, static_cast<int>(u0) - 4) & ~1, iu1 = std::min(W - 1, static_cast<int>(u1) + 4);
        int iv0 = std::max(0, static_cast<int>(v0) - 4), iv1 = std::min(H - 1, static_cast<int>(v1) + 4);
        if (iu0 > iu1 || iv0 > iv1 || (iu1 - iu0) * (iv1 - iv0) > W * H / 2) return;
        const V3 o = c.center;
        for (int v = iv0; v <= iv1; v++) {
            uint8_t *row = out + rowOffset(v);
            for (int u = iu0; u <= iu1 && u + 1 < W; u += 2) {
                uint8_t px[2][3]; bool changed = false;
                for (int s = 0; s < 2; s++) {
                    size_t i = static_cast<size_t>(v) * W + u + s;
                    std::memcpy(px[s], &pre_->rgb[i * 3], 3);
                    float sd = pre_->depth[i];
                    if (sd < 0) continue;
                    V3 d = {pre_->ray[i * 3], pre_->ray[i * 3 + 1], pre_->ray[i * 3 + 2]};
                    // ray / vertical cylinder
                    double ox = o.x - base.x, oy = o.y - base.y;
                    double A = d.x * d.x + d.y * d.y, B = 2 * (ox * d.x + oy * d.y), C = ox * ox + oy * oy - R * R;
                    double disc = B * B - 4 * A * C;
                    double t = -1;
                    if (A > 1e-12 && disc >= 0) {
                        double t0 = (-B - std::sqrt(disc)) / (2 * A);
                        double z = o.z + d.z * t0;
                        if (t0 > 0 && z <= 0 && z >= -HGT) t = t0;
                    }
                    if (t < 0 && d.z > 1e-9) {          // flat top cap
                        double tc = (-HGT - o.z) / d.z;
                        V3 p = o + d * tc;
                        if (tc > 0 && std::hypot(p.x - base.x, p.y - base.y) <= R) t = tc;
                    }
                    if (t > 0 && t < sd) {
                        V3 p = o + d * t;
                        double h = -p.z;
                        V3 n = norm({p.x - base.x, p.y - base.y, 0});
                        double lam = 0.5 + 0.5 * std::max(0.0, dot(n, SUN));
                        bool white = (h > 450 && h < 600) || (h > 700 && h < 820);
                        RGB col = white ? RGB{240, 240, 236} : RGB{245, 92, 18};
                        if (h >= HGT - 1) lam = 0.95;
                        px[s][0] = clamp8(col.r * lam); px[s][1] = clamp8(col.g * lam); px[s][2] = clamp8(col.b * lam);
                        changed = true;
                    } else if (d.z > 1e-9) {
                        // soft contact shadow of the post on the ground
                        double tg = -o.z / d.z;
                        if (std::fabs(tg - sd) < 5) {
                            V3 g = o + d * tg;
                            double r = std::hypot(g.x - base.x, g.y - base.y);
                            if (r < R * 2.2) {
                                double f = 0.55 + 0.45 * std::min(1.0, std::max(0.0, (r - R) / (R * 1.2)));
                                for (int k = 0; k < 3; k++) px[s][k] = clamp8(px[s][k] * f);
                                changed = true;
                            }
                        }
                    }
                }
                if (changed) writeUyvyPair(row + u * 2, px[0], px[1]);
            }
        }
    }

    void drawLabel(uint8_t *out) {
        static const char *names[4] = {"GAUCHE", "DROITE", "AVANT", "ARRIERE"};
        char text[48];
        time_t now = time(nullptr); struct tm tmv{}; localtime_r(&now, &tmv);
        snprintf(text, sizeof(text), "SIM %s %02d:%02d:%02d", names[slot_], tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
        const int scale = 2, len = static_cast<int>(std::strlen(text));
        const int tw = len * 6 * scale, x0 = (W - tw) / 2, y0 = 10;
        // The app mirrors the rear feed for the driver: pre-mirror its label so it reads normally.
        const bool mirror = slot_ == 3;
        static const uint8_t ink[3] = {255, 255, 255}, shade[3] = {10, 10, 10};
        for (int v = y0 - 2; v < y0 + 7 * scale + 2; v++) {
            uint8_t *row = out + rowOffset(v);
            for (int u = (x0 - 4) & ~1; u < x0 + tw + 4 && u + 1 < W; u += 2) {
                uint8_t px[2][3];
                for (int s = 0; s < 2; s++) {
                    int uu = u + s, lu = mirror ? (2 * x0 + tw - 1 - uu) : uu;
                    int gx = (lu - x0) / scale, gy = (v - y0) / scale;
                    bool on = false;
                    if (lu >= x0 && gx < len * 6 && gy >= 0 && gy < 7 && gx % 6 < 5)
                        on = (glyph(text[gx / 6])[gy] >> (4 - gx % 6)) & 1;
                    std::memcpy(px[s], on ? ink : shade, 3);
                }
                writeUyvyPair(row + u * 2, px[0], px[1]);
            }
        }
    }
};

} // namespace mg4sim
