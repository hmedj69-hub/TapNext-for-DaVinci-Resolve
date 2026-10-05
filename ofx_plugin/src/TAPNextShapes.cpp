// TAPNext Shapes — effet OpenFX pour DaVinci Resolve (Studio).
//
// Dessine des formes (cercle, étoile, image PNG…) sur les points suivis par
// TAPNext++ dans TAPNext Studio (fichier .tapfx) et produit une matte alpha.
// Tous les réglages sont dans l'Inspecteur de Resolve ; le rendu de chaque
// image est indépendant (le suivi, lui, est calculé une fois dans Studio).
//
// Format .tapfx (little endian) :
//   char[8] "TAPFX01\0" · int32 W, H, T, Q, first_frame, reserved[3] · float32 fps
//   float32 pos[T][Q][2] (pixels source, NaN = absent) · float32 vis[T][Q]
//   uint8 valid[T][Q] (suivi et non coupé) · int32 group[Q]
//   version 2 ("TAPFX02") : + float32 depth[T][Q] (0 = proche, 1 = loin, NaN = inconnue)
//   version 3 ("TAPFX03") : + float32 affine[T][Q][4] (déformation locale de la surface,
//                            déduite du suivi : perspective et profondeur des formes)
//
// Licence : MIT (code TAPNext) ; en-têtes OpenFX sous licence BSD.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxParam.h"
#include "ofxProperty.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_NO_STDIO_DEPRECATION
#define STBI_WINDOWS_UTF8
#include "stb_image.h"

#if defined(_WIN32)
#define TAP_EXPORT extern "C" __declspec(dllexport)
#else
#define TAP_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// Suites OFX
// ---------------------------------------------------------------------------
OfxHost* gHost = nullptr;
const OfxPropertySuiteV1* gProp = nullptr;
const OfxImageEffectSuiteV1* gEffect = nullptr;
const OfxParameterSuiteV1* gParam = nullptr;

const char* kPluginId = "com.tapnext.shapes";

// ---------------------------------------------------------------------------
// Paramètres (identifiant, libellé)
// ---------------------------------------------------------------------------
enum ShapeKind { kCircle, kSquare, kRounded, kDiamond, kTriangle, kHexagon, kStar, kCross, kRing, kImage };
const char* kShapeLabels[] = {"Cercle", "Carré", "Carré arrondi", "Losange", "Triangle",
                              "Hexagone", "Étoile", "Croix", "Anneau", "Image (PNG)"};
const char* kOutputLabels[] = {"Image + alpha (pour la page Color)", "Matte N&B",
                               "Aperçu (formes sur l'image)", "Rush masqué + effets"};
const char* kBgLabels[] = {"Transparent (alpha)", "Noir", "Rush original", "Rush assombri"};

struct Params {
  std::string file;
  int group = 0;
  int offset = 0;
  int output = 0;
  int shape = kCircle;
  std::string image;
  double size = 4.0, opacity = 1.0, rotation = 0.0, jitter = 0.0;
  bool follow = false;
  double grow = 0.1, stretch = 0.195, maxScale = 3.0;
  bool always = false, fadeInOn = false, fadeOutOn = false;
  int fadeIn = 6, fadeOut = 8;
  double merge = 0.10, threshold = 0.10, softness = 0.0;
  int trail = 0;
  double smooth = 2.1;
  bool invert = true;
  bool showPoints = false;
  double shapeW = 1.0, shapeH = 1.0;     // largeur / hauteur (× taille)
  bool perspOn = false;                  // épouser la surface (perspective du suivi)
  double persp = 1.0;
  int background = 0;                    // rush masqué : 0 transparent, 1 noir, 2 original, 3 assombri
  // profondeur
  double depthScale = 0.0, depthNear = 0.0, depthFar = 1.0, depthFeather = 0.05, depthFog = 0.0;
  // ombre portée
  bool shadowOn = false, shadowDepth = false;
  double shadowAngle = 135.0, shadowDist = 12.0, shadowSoft = 6.0, shadowOpacity = 0.6;
  // écho / slit-scan
  int echoMode = 0, echoCount = 4, echoStep = 3, slitSpan = 24;
  double echoDecay = 0.6, echoScale = 1.0;
};
enum EchoMode { kEchoNone, kEcho, kSlitH, kSlitV, kSlitRadial, kSlitDepth };
const char* kEchoLabels[] = {"Aucun", "Écho (copies dans le temps)", "Slit-scan horizontal",
                             "Slit-scan vertical", "Slit-scan radial", "Time-slice selon la profondeur (suivi)"};

// ---------------------------------------------------------------------------
// Fichiers (chemins UTF-8, y compris sous Windows : accents, etc.)
// ---------------------------------------------------------------------------
#if defined(_WIN32)
std::wstring widen(const std::string& s) {
  int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
  std::wstring w(n > 0 ? n : 1, L'\0');
  if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
  w.resize(wcslen(w.c_str()));
  return w;
}
std::string narrow(const wchar_t* w) {
  int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
  std::string s(n > 0 ? n : 1, '\0');
  if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, nullptr, nullptr);
  s.resize(std::strlen(s.c_str()));
  return s;
}
FILE* uopen(const std::string& path, const char* mode) {
  std::wstring m(mode, mode + std::strlen(mode));
  return _wfopen(widen(path).c_str(), m.c_str());
}
#else
FILE* uopen(const std::string& path, const char* mode) { return std::fopen(path.c_str(), mode); }
#endif

struct FileCloser { void operator()(FILE* f) const { if (f) std::fclose(f); } };
using FilePtr = std::unique_ptr<FILE, FileCloser>;

bool readAll(FILE* f, void* dst, size_t n) { return std::fread(dst, 1, n, f) == n; }

// ---------------------------------------------------------------------------
// Données de suivi
// ---------------------------------------------------------------------------
struct TrackFile {
  int W = 0, H = 0, T = 0, Q = 0, first = 0;
  float fps = 25.f;
  std::vector<float> pos, vis;
  std::vector<uint8_t> valid;
  std::vector<int32_t> group;
  std::vector<float> depth;   // vide si fichier v1
  std::vector<float> affine;  // T*Q*4 (vide avant v3)
};

bool loadTrackFile(const std::string& path, TrackFile& tf) {
  if (path.empty()) return false;
  FilePtr f(uopen(path, "rb"));
  if (!f) return false;
  char magic[8];
  int32_t hdr[8];
  if (!readAll(f.get(), magic, 8) || std::memcmp(magic, "TAPFX0", 6) != 0) return false;
  const int version = magic[6] - '0';
  if (!readAll(f.get(), hdr, sizeof(hdr)) || !readAll(f.get(), &tf.fps, 4)) return false;
  tf.W = hdr[0]; tf.H = hdr[1]; tf.T = hdr[2]; tf.Q = hdr[3]; tf.first = hdr[4];
  if (tf.W <= 0 || tf.H <= 0 || tf.T <= 0 || tf.Q <= 0 || tf.T > 1000000 || tf.Q > 1000000)
    return false;
  const size_t n = size_t(tf.T) * tf.Q;
  tf.pos.resize(n * 2); tf.vis.resize(n); tf.valid.resize(n); tf.group.resize(tf.Q);
  bool ok = readAll(f.get(), tf.pos.data(), n * 2 * 4) && readAll(f.get(), tf.vis.data(), n * 4) &&
            readAll(f.get(), tf.valid.data(), n) && readAll(f.get(), tf.group.data(), size_t(tf.Q) * 4);
  if (ok && version >= 2) {
    tf.depth.resize(n);
    if (!readAll(f.get(), tf.depth.data(), n * 4)) tf.depth.clear();
  }
  if (ok && version >= 3 && !tf.depth.empty()) {
    tf.affine.resize(n * 4);
    if (!readAll(f.get(), tf.affine.data(), n * 16)) tf.affine.clear();
  }
  return ok;
}

// Données préparées pour un groupe (dépendent du lissage et des fondus).
struct Prepared {
  int T = 0, n = 0;
  std::vector<float> pos;    // T*n*2 positions de dessin
  std::vector<float> alpha;  // T*n
  std::vector<float> speed;  // T*n
  std::vector<float> dir;    // T*n*2
  std::vector<float> jit;    // n
  std::vector<float> depth;  // T*n (vide si pas de profondeur)
  std::vector<float> affine; // T*n*4 (vide si pas de perspective)
};

inline bool finite(float v) { return std::isfinite(v); }

std::vector<float> gaussKernel(double sigma) {
  int r = std::max(1, int(std::ceil(3 * sigma)));
  std::vector<float> k(2 * r + 1);
  double s = 0;
  for (int i = -r; i <= r; ++i) { k[i + r] = float(std::exp(-0.5 * (i / sigma) * (i / sigma))); s += k[i + r]; }
  for (auto& v : k) v = float(v / s);
  return k;
}

// Portage de tap_resolve_tool.py / shape_engine.py (prepare_group).
void prepare(const TrackFile& tf, const Params& p, Prepared& out) {
  std::vector<int> cols;
  for (int q = 0; q < tf.Q; ++q)
    if (p.group <= 0 || tf.group[q] == p.group - 1) cols.push_back(q);
  const int T = tf.T, n = int(cols.size());
  out.T = T; out.n = n;
  auto P = [&](int t, int j, int c) { return tf.pos[(size_t(t) * tf.Q + cols[j]) * 2 + c]; };
  auto V = [&](int t, int j) { return tf.vis[size_t(t) * tf.Q + cols[j]]; };
  auto OK = [&](int t, int j) { return tf.valid[size_t(t) * tf.Q + cols[j]] != 0; };

  // 1) lissage gaussien pondéré par la visibilité (convolution normalisée)
  std::vector<float> sm(size_t(T) * n * 2);
  for (int t = 0; t < T; ++t)
    for (int j = 0; j < n; ++j) { sm[(size_t(t) * n + j) * 2] = P(t, j, 0); sm[(size_t(t) * n + j) * 2 + 1] = P(t, j, 1); }
  if (p.smooth > 0) {
    auto k = gaussKernel(p.smooth);
    const int r = int(k.size() / 2);
    for (int j = 0; j < n; ++j)
      for (int t = 0; t < T; ++t) {
        if (!finite(P(t, j, 0))) continue;
        double num0 = 0, num1 = 0, den = 0;
        for (int i = -r; i <= r; ++i) {
          int u = t + i;
          if (u < 0 || u >= T || !finite(P(u, j, 0))) continue;
          double w = k[i + r] * std::max(1e-3f, V(u, j));
          num0 += w * P(u, j, 0); num1 += w * P(u, j, 1); den += w;
        }
        if (den > 1e-4) { sm[(size_t(t) * n + j) * 2] = float(num0 / den); sm[(size_t(t) * n + j) * 2 + 1] = float(num1 / den); }
      }
  }
  // 2) vitesse (différence centrée)
  out.dir.assign(size_t(T) * n * 2, 0.f);
  out.speed.assign(size_t(T) * n, 0.f);
  for (int j = 0; j < n; ++j)
    for (int t = 0; t < T; ++t) {
      int a = std::max(0, t - 1), b = std::min(T - 1, t + 1);
      if (a == b) continue;
      float dx = (sm[(size_t(b) * n + j) * 2] - sm[(size_t(a) * n + j) * 2]) / float(b - a);
      float dy = (sm[(size_t(b) * n + j) * 2 + 1] - sm[(size_t(a) * n + j) * 2 + 1]) / float(b - a);
      if (!finite(dx) || !finite(dy)) dx = dy = 0;
      out.dir[(size_t(t) * n + j) * 2] = dx;
      out.dir[(size_t(t) * n + j) * 2 + 1] = dy;
      out.speed[size_t(t) * n + j] = std::sqrt(dx * dx + dy * dy);
    }
  // 3) opacité + positions de dessin
  out.alpha.assign(size_t(T) * n, 0.f);
  out.pos = sm;
  if (p.always) {
    for (int t = 0; t < T; ++t)
      for (int j = 0; j < n; ++j) {
        bool ok = OK(t, j) && finite(sm[(size_t(t) * n + j) * 2]);
        out.alpha[size_t(t) * n + j] = ok ? 1.f : 0.f;
        if (!ok) out.speed[size_t(t) * n + j] = 0;
      }
  } else {
    const float up = 1.f / std::max(1, p.fadeInOn ? p.fadeIn : 1);
    const float down = 1.f / std::max(1, p.fadeOutOn ? p.fadeOut : 1);
    for (int j = 0; j < n; ++j) {
      float a = std::min(1.f, std::max(0.f, (V(0, j) - 0.4f) / 0.2f));
      float lx = NAN, ly = NAN;
      for (int t = 0; t < T; ++t) {
        float target = std::min(1.f, std::max(0.f, (V(t, j) - 0.4f) / 0.2f));
        a = target > a ? std::min(target, a + up) : std::max(target, a - down);
        out.alpha[size_t(t) * n + j] = a;
        size_t i2 = (size_t(t) * n + j) * 2;
        bool vis = V(t, j) >= 0.5f && finite(sm[i2]);
        if (vis) { lx = sm[i2]; ly = sm[i2 + 1]; }
        else {
          out.speed[size_t(t) * n + j] = 0;
          if (finite(lx)) { out.pos[i2] = lx; out.pos[i2 + 1] = ly; }
        }
      }
      // avant la première apparition : première position visible
      float nx = NAN, ny = NAN;
      bool seen = false;
      std::vector<bool> seenBefore(T, false);
      for (int t = 0; t < T; ++t) { if (V(t, j) >= 0.5f && finite(sm[(size_t(t) * n + j) * 2])) seen = true; seenBefore[t] = seen; }
      for (int t = T - 1; t >= 0; --t) {
        size_t i2 = (size_t(t) * n + j) * 2;
        if (V(t, j) >= 0.5f && finite(sm[i2])) { nx = sm[i2]; ny = sm[i2 + 1]; }
        if (!seenBefore[t] && finite(nx)) { out.pos[i2] = nx; out.pos[i2 + 1] = ny; }
      }
    }
  }
  // 3b) profondeur maintenue pendant les occultations
  if (!tf.depth.empty()) {
    out.depth.assign(size_t(T) * n, NAN);
    for (int j = 0; j < n; ++j) {
      float last = NAN, firstv = NAN;
      for (int t = 0; t < T; ++t) {
        float d = tf.depth[size_t(t) * tf.Q + cols[j]];
        if (finite(d)) { last = d; if (!finite(firstv)) firstv = d; }
        out.depth[size_t(t) * n + j] = last;
      }
      for (int t = 0; t < T && !finite(out.depth[size_t(t) * n + j]); ++t) out.depth[size_t(t) * n + j] = firstv;
    }
  }
  // 3c) perspective : déformation locale (déjà complétée par Studio)
  if (!tf.affine.empty()) {
    out.affine.resize(size_t(T) * n * 4);
    for (int t = 0; t < T; ++t)
      for (int j = 0; j < n; ++j)
        for (int c = 0; c < 4; ++c) {
          float v = tf.affine[(size_t(t) * tf.Q + cols[j]) * 4 + c];
          out.affine[(size_t(t) * n + j) * 4 + c] = finite(v) ? v : (c == 0 || c == 3 ? 1.f : 0.f);
        }
  }
  // 4) variation de taille déterministe par point
  out.jit.resize(n);
  for (int j = 0; j < n; ++j) {
    uint32_t h = uint32_t(cols[j]) * 2654435761u + 12345u;
    h ^= h >> 13; h *= 0x5bd1e995; h ^= h >> 15;
    float r01 = (h & 0xFFFFFF) / float(0xFFFFFF);
    out.jit[j] = float(1.0 + p.jitter * (r01 * 2 - 1));
  }
}

// ---------------------------------------------------------------------------
// Formes : fonctions de distance (rayon unité) → anticrénelage analytique
// ---------------------------------------------------------------------------
struct Poly { std::vector<float> nx, ny, d; };  // demi-plans n·p <= d

Poly convexPoly(const std::vector<std::pair<float, float>>& v) {
  Poly P;
  const size_t m = v.size();
  double area = 0;
  for (size_t i = 0; i < m; ++i) {
    auto a = v[i], b = v[(i + 1) % m];
    area += a.first * b.second - b.first * a.second;
  }
  float s = area > 0 ? 1.f : -1.f;
  for (size_t i = 0; i < m; ++i) {
    auto a = v[i], b = v[(i + 1) % m];
    float ex = b.first - a.first, ey = b.second - a.second;
    float nx = s * ey, ny = -s * ex;
    float L = std::sqrt(nx * nx + ny * ny);
    nx /= L; ny /= L;
    P.nx.push_back(nx); P.ny.push_back(ny); P.d.push_back(nx * a.first + ny * a.second);
  }
  return P;
}

inline float polySdf(const Poly& P, float x, float y) {
  float m = -1e9f;
  for (size_t i = 0; i < P.d.size(); ++i) m = std::max(m, P.nx[i] * x + P.ny[i] * y - P.d[i]);
  return m;
}

std::vector<std::pair<float, float>> regular(int n, float r, float phaseDeg) {
  std::vector<std::pair<float, float>> v;
  for (int i = 0; i < n; ++i) {
    double a = (phaseDeg + i * 360.0 / n) * kPi / 180.0;
    v.push_back({float(r * std::cos(a)), float(r * std::sin(a))});
  }
  return v;
}

struct ShapeLib {
  Poly triangle, hexagon, diamond, pent;
  std::vector<Poly> starTips;
  ShapeLib() {
    triangle = convexPoly(regular(3, 1.2f, -90));
    hexagon = convexPoly(regular(6, 1.0f, 0));
    diamond = convexPoly({{0, -1.2f}, {1.2f, 0}, {0, 1.2f}, {-1.2f, 0}});
    auto outer = regular(5, 1.2f, -90), inner = regular(5, 0.54f, -54);
    pent = convexPoly(inner);
    for (int i = 0; i < 5; ++i)
      starTips.push_back(convexPoly({inner[(i + 4) % 5], outer[i], inner[i]}));
  }
} const* gShapes = nullptr;

struct Stamp {
  std::vector<float> a;  // valeurs 0..1
  int w = 0, h = 0;
};

inline float boxSdf(float x, float y, float bx, float by) {
  float qx = std::fabs(x) - bx, qy = std::fabs(y) - by;
  float ox = std::max(qx, 0.f), oy = std::max(qy, 0.f);
  return std::sqrt(ox * ox + oy * oy) + std::min(std::max(qx, qy), 0.f);
}

// Distance signée (unités : rayon) d'un point local à la forme.
inline float shapeSdf(int kind, float x, float y) {
  switch (kind) {
    case kSquare: return boxSdf(x, y, 0.886f, 0.886f);
    case kRounded: return boxSdf(x, y, 0.58f, 0.58f) - 0.35f;
    case kDiamond: return polySdf(gShapes->diamond, x, y);
    case kTriangle: return polySdf(gShapes->triangle, x, y);
    case kHexagon: return polySdf(gShapes->hexagon, x, y);
    case kStar: {
      float d = polySdf(gShapes->pent, x, y);
      for (auto& tip : gShapes->starTips) d = std::min(d, polySdf(tip, x, y));
      return d;
    }
    case kCross: return std::min(boxSdf(x, y, 0.34f, 1.f), boxSdf(x, y, 1.f, 0.34f));
    case kRing: return std::fabs(std::sqrt(x * x + y * y) - 1.f) - 0.15f;
    default: return std::sqrt(x * x + y * y) - 1.f;
  }
}

// ---------------------------------------------------------------------------
// Rendu d'une matte à la résolution de travail
// ---------------------------------------------------------------------------
struct Canvas {
  int w = 0, h = 0;
  std::vector<float> v;
  void init(int W, int H) { w = W; h = H; v.assign(size_t(W) * H, 0.f); }
};

// M = matrice 2×2 forme unitaire → pixels (taille, rotation, étirement, perspective).
void drawShape(Canvas& c, const Params& p, const Stamp* stamp, float cx, float cy, const float* M,
               float value) {
  const float det = M[0] * M[3] - M[1] * M[2];
  if (std::fabs(det) < 1e-6f) return;
  const float i00 = M[3] / det, i01 = -M[1] / det, i10 = -M[2] / det, i11 = M[0] / det;
  const float R = std::sqrt(M[0] * M[0] + M[1] * M[1] + M[2] * M[2] + M[3] * M[3]) * 1.3f + 2.f;
  const int x0 = std::max(0, int(cx - R)), x1 = std::min(c.w - 1, int(cx + R));
  const int y0 = std::max(0, int(cy - R)), y1 = std::min(c.h - 1, int(cy + R));
  const float scale = std::max(0.5f, std::sqrt(std::fabs(det)));
  // similitude (rotation + échelle uniforme) ou forme déformée ?
  const bool aniso = std::fabs(M[0] - M[3]) + std::fabs(M[1] + M[2]) > 0.05f * (std::fabs(M[0]) + std::fabs(M[1]) + std::fabs(M[2]) + std::fabs(M[3]));
  for (int y = y0; y <= y1; ++y) {
    float* row = &c.v[size_t(y) * c.w];
    for (int x = x0; x <= x1; ++x) {
      float dx = x + 0.5f - cx, dy = y + 0.5f - cy;
      float lx = i00 * dx + i01 * dy;
      float ly = i10 * dx + i11 * dy;
      float cov;
      if (p.shape == kImage && stamp && stamp->w > 0) {
        float u = (lx * 0.5f + 0.5f) * (stamp->w - 1), vv = (ly * 0.5f + 0.5f) * (stamp->h - 1);
        if (u < 0 || vv < 0 || u > stamp->w - 1 || vv > stamp->h - 1) continue;
        int iu = int(u), iv = int(vv);
        int iu1 = std::min(iu + 1, stamp->w - 1), iv1 = std::min(iv + 1, stamp->h - 1);
        float fu = u - iu, fv = vv - iv;
        const float* s = stamp->a.data();
        cov = (s[iv * stamp->w + iu] * (1 - fu) + s[iv * stamp->w + iu1] * fu) * (1 - fv) +
              (s[iv1 * stamp->w + iu] * (1 - fu) + s[iv1 * stamp->w + iu1] * fu) * fv;
      } else {
        const int kind = p.shape == kImage ? kCircle : p.shape;
        float d = shapeSdf(kind, lx, ly);
        if (aniso) {
          // Forme étirée : distance en pixels via le gradient (bords nets partout).
          const float e = 0.5f;
          float gx = (shapeSdf(kind, lx + i00 * e, ly + i10 * e) - d) / e;
          float gy = (shapeSdf(kind, lx + i01 * e, ly + i11 * e) - d) / e;
          float g = std::sqrt(gx * gx + gy * gy);
          d = g > 1e-6f ? d / g : d * scale;
        } else {
          d *= scale;
        }
        cov = std::min(1.f, std::max(0.f, 0.5f - d));
      }
      float val = cov * value;
      if (val > row[x]) row[x] = val;
    }
  }
}

void gaussBlur(Canvas& c, double sigma) {
  if (sigma <= 0.3) return;
  // Réduction si le flou est grand (même stratégie que shape_engine.py).
  int f = std::max(1, int(sigma / 4.0));
  int w = (c.w + f - 1) / f, h = (c.h + f - 1) / f;
  std::vector<float> s(size_t(w) * h, 0.f);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double acc = 0; int cnt = 0;
      for (int yy = y * f; yy < std::min(c.h, (y + 1) * f); ++yy)
        for (int xx = x * f; xx < std::min(c.w, (x + 1) * f); ++xx) { acc += c.v[size_t(yy) * c.w + xx]; ++cnt; }
      s[size_t(y) * w + x] = cnt ? float(acc / cnt) : 0.f;
    }
  auto k = gaussKernel(sigma / f);
  const int r = int(k.size() / 2);
  std::vector<float> tmp(s.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double acc = 0;
      for (int i = -r; i <= r; ++i) { int xx = std::min(w - 1, std::max(0, x + i)); acc += k[i + r] * s[size_t(y) * w + xx]; }
      tmp[size_t(y) * w + x] = float(acc);
    }
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double acc = 0;
      for (int i = -r; i <= r; ++i) { int yy = std::min(h - 1, std::max(0, y + i)); acc += k[i + r] * tmp[size_t(yy) * w + x]; }
      s[size_t(y) * w + x] = float(acc);
    }
  if (f == 1) { c.v.swap(s); return; }
  for (int y = 0; y < c.h; ++y)
    for (int x = 0; x < c.w; ++x) {
      float fx = (x + 0.5f) / f - 0.5f, fy = (y + 0.5f) / f - 0.5f;
      int ix = std::max(0, std::min(w - 1, int(std::floor(fx)))), iy = std::max(0, std::min(h - 1, int(std::floor(fy))));
      int ix1 = std::min(w - 1, ix + 1), iy1 = std::min(h - 1, iy + 1);
      float ax = std::min(1.f, std::max(0.f, fx - ix)), ay = std::min(1.f, std::max(0.f, fy - iy));
      c.v[size_t(y) * c.w + x] = (s[size_t(iy) * w + ix] * (1 - ax) + s[size_t(iy) * w + ix1] * ax) * (1 - ay) +
                                 (s[size_t(iy1) * w + ix] * (1 - ax) + s[size_t(iy1) * w + ix1] * ax) * ay;
    }
}

// Matte du temps t (index dans le fichier), à la résolution de travail ww×wh.
// Matte d'un groupe à l'instant t (sans écho, ombre, opacité ni inversion).
void renderCore(const TrackFile& tf, const Prepared& pr, const Params& p, const Stamp* stamp,
                int t, int ww, int wh, float scale, bool shadow, std::vector<float>& out) {
  const float rs = float(ww) / tf.W;            // pixels de travail par pixel source
  const float rsy = float(wh) / tf.H;
  const float vnorm = 1080.f / tf.H;
  const float base = std::max(0.5f, float(p.size) * rs * scale);
  const bool hasDepth = !pr.depth.empty();
  struct S { float cx, cy, m[4], al; };
  const bool usePersp = p.perspOn && !pr.affine.empty();
  std::vector<S> st;
  const int trail = std::max(0, p.trail);
  const float ca = std::cos(float(p.shadowAngle * kPi / 180)), sa = std::sin(float(p.shadowAngle * kPi / 180));
  for (int k = 0; k <= trail; ++k) {
    float fade = 1.f - k / (trail + 1.f);
    for (int j = 0; j < pr.n; ++j) {
      int t0 = t;
      if (p.echoMode == kSlitDepth && hasDepth && t >= 0 && t < pr.T) {
        float d0 = pr.depth[size_t(t) * pr.n + j];
        if (finite(d0)) t0 = t - int(std::lround(d0 * std::max(1, p.slitSpan)));
      }
      int tk = t0 - k;
      if (tk < 0 || tk >= pr.T) continue;
      size_t i = size_t(tk) * pr.n + j;
      float al = pr.alpha[i];
      float x = pr.pos[i * 2], y = pr.pos[i * 2 + 1];
      if (al <= 1.f / 255 || !finite(x)) continue;
      float v = pr.speed[i] * vnorm;
      float g = std::min(float(p.maxScale), 1.f + float(p.grow) * v);
      float s = std::min(float(p.maxScale), 1.f + float(p.stretch) * v);
      float r = base * pr.jit[j] * g * (k ? fade : 1.f);
      float d = hasDepth ? pr.depth[i] : NAN;
      if (finite(d)) {
        if (p.depthScale > 0) r *= float((1 - p.depthScale) + p.depthScale * (1.6 - 1.2 * d));
        if (p.depthNear > 0 || p.depthFar < 1) {
          float fe = float(std::max(1e-3, p.depthFeather));
          float ain = std::min(1.f, std::max(0.f, (d - float(p.depthNear - fe)) / fe));
          float aout = std::min(1.f, std::max(0.f, (float(p.depthFar + fe) - d) / fe));
          al *= ain * aout;
        }
        if (p.depthFog > 0) al *= float(1.0 - p.depthFog * d);
      }
      if (al <= 1.f / 255) continue;
      float ang = float(p.rotation);
      if ((p.follow || p.stretch > 0) && pr.speed[i] > 0.05f)
        ang += std::atan2(pr.dir[i * 2 + 1], pr.dir[i * 2]) * 180.f / float(kPi);
      float cx = x * rs, cy = y * rsy;
      if (shadow) {
        float dist = float(p.shadowDist) * rs;
        if (p.shadowDepth && finite(d)) dist *= 0.25f + 1.5f * (1.f - d);
        cx += ca * dist; cy += sa * dist;
      }
      {
        const float sxx = std::max(0.5f, r * s * float(p.shapeW)), syy = std::max(0.5f, r * float(p.shapeH));
        const float ra = ang * float(kPi) / 180.f, cr = std::cos(ra), sr = std::sin(ra);
        S e{cx, cy, {cr * sxx, -sr * syy, sr * sxx, cr * syy}, al * (k ? fade : 1.f)};
        if (usePersp) {
          const float* A = &pr.affine[i * 4];
          const float am = float(p.persp);
          const float P0 = 1 + am * (A[0] - 1), P1 = am * A[1], P2 = am * A[2], P3 = 1 + am * (A[3] - 1);
          const float m0 = e.m[0], m1 = e.m[1], m2 = e.m[2], m3 = e.m[3];
          e.m[0] = P0 * m0 + P1 * m2; e.m[1] = P0 * m1 + P1 * m3;
          e.m[2] = P2 * m0 + P3 * m2; e.m[3] = P2 * m1 + P3 * m3;
        }
        st.push_back(e);
      }
    }
  }
  Canvas m;
  m.init(ww, wh);
  if (p.merge > 0) {
    Canvas amap, cover;
    amap.init(ww, wh); cover.init(ww, wh);
    for (auto& s : st) {
      const float big[4] = {s.m[0] * 1.5f, s.m[1] * 1.5f, s.m[2] * 1.5f, s.m[3] * 1.5f};
      drawShape(m, p, stamp, s.cx, s.cy, s.m, 1.f);
      drawShape(amap, p, stamp, s.cx, s.cy, big, s.al);
      drawShape(cover, p, stamp, s.cx, s.cy, big, 1.f);
    }
    double sigma = p.merge * base;
    gaussBlur(m, sigma); gaussBlur(amap, sigma); gaussBlur(cover, sigma);
    float lo = float(p.threshold - p.softness), hi = float(p.threshold + p.softness);
    for (size_t i = 0; i < m.v.size(); ++i) {
      float f = m.v[i], v;
      if (hi - lo < 1e-4f) v = f >= p.threshold ? 1.f : 0.f;
      else { v = std::min(1.f, std::max(0.f, (f - lo) / (hi - lo))); v = v * v * (3 - 2 * v); }
      float op = std::min(1.f, std::max(0.f, amap.v[i] / std::max(cover.v[i], 1e-3f)));
      m.v[i] = v * op;
    }
  } else {
    for (auto& s : st) drawShape(m, p, stamp, s.cx, s.cy, s.m, s.al);
    gaussBlur(m, p.softness * base * 2.0);
  }
  out.swap(m.v);
}

// Écho temporel / slit-scan.
void renderTimed(const TrackFile& tf, const Prepared& pr, const Params& p, const Stamp* stamp,
                 int t, int ww, int wh, std::vector<float>& out) {
  if (p.echoMode == kEcho) {
    renderCore(tf, pr, p, stamp, t, ww, wh, 1.f, false, out);
    std::vector<float> mk;
    for (int k = 1; k <= std::max(0, p.echoCount); ++k) {
      int tk = t - k * std::max(1, p.echoStep);
      if (tk < 0) break;
      renderCore(tf, pr, p, stamp, tk, ww, wh, float(std::pow(p.echoScale, k)), false, mk);
      float dec = float(std::pow(p.echoDecay, k));
      for (size_t i = 0; i < out.size(); ++i) out[i] = std::max(out[i], mk[i] * dec);
    }
    return;
  }
  if (p.echoMode == kSlitH || p.echoMode == kSlitV || p.echoMode == kSlitRadial) {
    const int span = std::max(1, p.slitSpan);
    const int K = std::min(12, span + 1);
    std::vector<std::vector<float>> layers(K);
    for (int i = 0; i < K; ++i) {
      int tk = std::max(0, t - int(std::lround(double(i) * span / (K - 1))));
      renderCore(tf, pr, p, stamp, tk, ww, wh, 1.f, false, layers[i]);
    }
    out.assign(size_t(ww) * wh, 0.f);
    const float cx = (ww - 1) / 2.f, cy = (wh - 1) / 2.f, rmax = std::sqrt(cx * cx + cy * cy);
    for (int y = 0; y < wh; ++y)
      for (int x = 0; x < ww; ++x) {
        float f = p.echoMode == kSlitH ? x / float(std::max(1, ww - 1))
                : p.echoMode == kSlitV ? y / float(std::max(1, wh - 1))
                : std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy)) / rmax;
        float fi = f * (K - 1);
        int i0 = std::max(0, std::min(K - 1, int(std::floor(fi)))), i1 = std::min(K - 1, i0 + 1);
        float a = fi - i0;
        size_t idx = size_t(y) * ww + x;
        out[idx] = layers[i0][idx] * (1 - a) + layers[i1][idx] * a;
      }
    return;
  }
  renderCore(tf, pr, p, stamp, t, ww, wh, 1.f, false, out);
}

// Matte finale d'un temps t : écho/slit-scan, ombre portée, opacité, inversion.
void renderMatte(const TrackFile& tf, const Prepared& pr, const Params& p, const Stamp* stamp,
                 int t, int ww, int wh, std::vector<float>& out) {
  renderTimed(tf, pr, p, stamp, t, ww, wh, out);
  if (p.shadowOn && p.shadowOpacity > 0) {
    const float rs = float(ww) / tf.W;
    Canvas sh;
    sh.init(ww, wh);
    if (p.shadowDepth && !pr.depth.empty()) {
      renderCore(tf, pr, p, stamp, t, ww, wh, 1.f, true, sh.v);
    } else {
      float dist = float(p.shadowDist) * rs;
      int dx = int(std::lround(std::cos(p.shadowAngle * kPi / 180) * dist));
      int dy = int(std::lround(std::sin(p.shadowAngle * kPi / 180) * dist));
      for (int y = 0; y < wh; ++y)
        for (int x = 0; x < ww; ++x) {
          int sx = x - dx, sy = y - dy;
          if (sx >= 0 && sx < ww && sy >= 0 && sy < wh) sh.v[size_t(y) * ww + x] = out[size_t(sy) * ww + sx];
        }
    }
    gaussBlur(sh, p.shadowSoft * rs);
    const float so = float(p.shadowOpacity);
    for (size_t i = 0; i < out.size(); ++i) out[i] = out[i] + sh.v[i] * so * (1.f - out[i]);
  }
  const float op = float(p.opacity);
  for (auto& v : out) { v *= op; if (p.invert) v = 1.f - v; }
}

// ---------------------------------------------------------------------------
// Instance
// ---------------------------------------------------------------------------
struct Instance {
  OfxImageEffectHandle effect = nullptr;
  OfxImageClipHandle src = nullptr, dst = nullptr;
  std::map<std::string, OfxParamHandle> params;
  std::mutex mtx;
  // cache
  std::string loadedPath;
  long long loadedStamp = -1;
  std::shared_ptr<TrackFile> tf;
  std::string prepKey;
  std::shared_ptr<Prepared> prep;
  std::string stampPath;
  std::shared_ptr<Stamp> stamp;
};

Instance* getInstance(OfxImageEffectHandle effect) {
  OfxPropertySetHandle props;
  gEffect->getPropertySet(effect, &props);
  void* p = nullptr;
  gProp->propGetPointer(props, kOfxPropInstanceData, 0, &p);
  return static_cast<Instance*>(p);
}

std::string lastPathFrom(const char* fileName) {
  std::string dir;
#if defined(_WIN32)
  const wchar_t* a = _wgetenv(L"APPDATA");
  if (a) dir = narrow(a) + "\\TAPNext\\";
#elif defined(__APPLE__)
  const char* h = std::getenv("HOME");
  if (h) dir = std::string(h) + "/Library/Application Support/TAPNext/";
#else
  const char* h = std::getenv("HOME");
  if (h) dir = std::string(h) + "/.config/TAPNext/";
#endif
  if (dir.empty()) return "";
  FilePtr f(uopen(dir + fileName, "rb"));
  if (!f) return "";
  char buf[4096];
  size_t n = std::fread(buf, 1, sizeof(buf) - 1, f.get());
  std::string line(buf, n);
  if (line.size() >= 3 && (unsigned char)line[0] == 0xEF) line = line.substr(3);  // BOM
  while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) line.pop_back();
  return line;
}

std::string lastExportPath() { return lastPathFrom("last_tapfx.txt"); }

// Taille du fichier : détecte un nouvel export au même chemin.
long long fileStamp(const std::string& path) {
  FilePtr f(uopen(path, "rb"));
  if (!f) return -1;
  std::fseek(f.get(), 0, SEEK_END);
  return (long long)std::ftell(f.get());
}

// ---------------------------------------------------------------------------
// Définition des paramètres
// ---------------------------------------------------------------------------
OfxPropertySetHandle defParam(OfxParamSetHandle ps, const char* type, const char* name,
                              const char* label, const char* hint, const char* parent) {
  OfxPropertySetHandle props = nullptr;
  gParam->paramDefine(ps, type, name, &props);
  gProp->propSetString(props, kOfxPropLabel, 0, label);
  if (hint) gProp->propSetString(props, kOfxParamPropHint, 0, hint);
  if (parent) gProp->propSetString(props, kOfxParamPropParent, 0, parent);
  return props;
}

void defDouble(OfxParamSetHandle ps, const char* name, const char* label, double def, double lo,
               double hi, const char* parent, const char* hint = nullptr) {
  auto pr = defParam(ps, kOfxParamTypeDouble, name, label, hint, parent);
  gProp->propSetDouble(pr, kOfxParamPropDefault, 0, def);
  gProp->propSetDouble(pr, kOfxParamPropMin, 0, lo);
  gProp->propSetDouble(pr, kOfxParamPropMax, 0, hi);
  gProp->propSetDouble(pr, kOfxParamPropDisplayMin, 0, lo);
  gProp->propSetDouble(pr, kOfxParamPropDisplayMax, 0, hi);
  gProp->propSetInt(pr, kOfxParamPropDigits, 0, 3);
}

void defInt(OfxParamSetHandle ps, const char* name, const char* label, int def, int lo, int hi,
            const char* parent, const char* hint = nullptr) {
  auto pr = defParam(ps, kOfxParamTypeInteger, name, label, hint, parent);
  gProp->propSetInt(pr, kOfxParamPropDefault, 0, def);
  gProp->propSetInt(pr, kOfxParamPropMin, 0, lo);
  gProp->propSetInt(pr, kOfxParamPropMax, 0, hi);
  gProp->propSetInt(pr, kOfxParamPropDisplayMin, 0, lo);
  gProp->propSetInt(pr, kOfxParamPropDisplayMax, 0, hi);
}

void defBool(OfxParamSetHandle ps, const char* name, const char* label, bool def,
             const char* parent, const char* hint = nullptr) {
  auto pr = defParam(ps, kOfxParamTypeBoolean, name, label, hint, parent);
  gProp->propSetInt(pr, kOfxParamPropDefault, 0, def ? 1 : 0);
}

void defChoice(OfxParamSetHandle ps, const char* name, const char* label,
               const char* const* options, int n, int def, const char* parent) {
  auto pr = defParam(ps, kOfxParamTypeChoice, name, label, nullptr, parent);
  for (int i = 0; i < n; ++i) gProp->propSetString(pr, kOfxParamPropChoiceOption, i, options[i]);
  gProp->propSetInt(pr, kOfxParamPropDefault, 0, def);
}

void defString(OfxParamSetHandle ps, const char* name, const char* label, bool filePath,
               const char* parent, const char* hint) {
  auto pr = defParam(ps, kOfxParamTypeString, name, label, hint, parent);
  gProp->propSetString(pr, kOfxParamPropStringMode, 0,
                       filePath ? kOfxParamStringIsFilePath : kOfxParamStringIsSingleLine);
  if (filePath) gProp->propSetInt(pr, kOfxParamPropStringFilePathExists, 0, 1);
  gProp->propSetString(pr, kOfxParamPropDefault, 0, "");
}

void defGroup(OfxParamSetHandle ps, const char* name, const char* label, bool open = true) {
  auto pr = defParam(ps, kOfxParamTypeGroup, name, label, nullptr, nullptr);
  gProp->propSetInt(pr, kOfxParamPropGroupOpen, 0, open ? 1 : 0);
}

const char* kParamNames[] = {"file", "group", "offset", "output", "showPoints", "shape", "image",
                             "size", "shapeW", "shapeH", "opacity", "rotation", "follow", "jitter", "grow", "stretch",
                             "maxScale", "always", "fadeInOn", "fadeIn", "fadeOutOn", "fadeOut",
                             "merge", "threshold", "softness", "trail", "smooth", "invert",
                             "perspOn", "persp", "background",
                             "shadowOn", "shadowAngle", "shadowDist", "shadowSoft", "shadowOpacity",
                             "shadowDepth", "echoMode", "echoCount", "echoStep", "echoDecay",
                             "echoScale", "slitSpan"};

void describeParams(OfxImageEffectHandle desc) {
  OfxParamSetHandle ps;
  gEffect->getParamSet(desc, &ps);
  defGroup(ps, "gData", "Suivi TAPNext");
  defString(ps, "file", "Fichier de suivi (.tapfx)", true, "gData",
            "Fichier exporté par TAPNext Studio. Vide = dernier export.");
  defInt(ps, "group", "Groupe (0 = tous)", 0, 0, 999, "gData",
         "Numéro du groupe de points de Studio (0 = tous les points)");
  defInt(ps, "offset", "Décalage d'image", 0, -100000, 100000, "gData",
         "À ajuster si les formes sont en avance ou en retard sur l'image");
  defChoice(ps, "output", "Sortie", kOutputLabels, 4, 0, "gData");
  defChoice(ps, "background", "Fond du rush masqué", kBgLabels, 4, 0, "gData");
  defBool(ps, "showPoints", "Afficher les points (aperçu)", false, "gData");

  defGroup(ps, "gShape", "Forme");
  defChoice(ps, "shape", "Forme", kShapeLabels, 10, kCircle, "gShape");
  defString(ps, "image", "Image de forme (PNG)", true, "gShape",
            "Pour la forme « Image » : la transparence (ou la luminance) sert de forme");
  defDouble(ps, "size", "Taille (px)", 4.0, 0.5, 400.0, "gShape", "Rayon en pixels de la vidéo source");
  defDouble(ps, "shapeW", "Largeur (×)", 1.0, 0.05, 10.0, "gShape",
            "Étire la forme en largeur (rectangle, ellipse…)");
  defDouble(ps, "shapeH", "Hauteur (×)", 1.0, 0.05, 10.0, "gShape",
            "Étire la forme en hauteur");
  defDouble(ps, "opacity", "Opacité", 1.0, 0.0, 1.0, "gShape");
  defDouble(ps, "rotation", "Rotation (°)", 0.0, -180.0, 180.0, "gShape");
  defBool(ps, "follow", "Orienter dans le sens du mouvement", false, "gShape");
  defDouble(ps, "jitter", "Variation aléatoire de taille", 0.0, 0.0, 1.0, "gShape");
  defBool(ps, "perspOn", "Épouser la perspective de la surface", false, "gShape",
          "Les formes se déforment et changent de taille comme la surface suivie (déduit du suivi TAPNext)");
  defDouble(ps, "persp", "Intensité de la perspective", 1.0, 0.0, 1.0, "gShape");

  defGroup(ps, "gMotion", "Réaction au mouvement");
  defDouble(ps, "grow", "Grossir avec la vitesse", 0.1, 0.0, 1.0, "gMotion");
  defDouble(ps, "stretch", "Étirer dans la direction", 0.195, 0.0, 1.0, "gMotion");
  defDouble(ps, "maxScale", "Agrandissement maximal (×)", 3.0, 1.0, 8.0, "gMotion");

  defGroup(ps, "gVis", "Apparition");
  defBool(ps, "always", "Toujours visible (ignorer les occultations)", false, "gVis");
  defBool(ps, "fadeInOn", "Fondu d'apparition", false, "gVis");
  defInt(ps, "fadeIn", "Durée apparition (images)", 6, 1, 120, "gVis");
  defBool(ps, "fadeOutOn", "Fondu de disparition", false, "gVis");
  defInt(ps, "fadeOut", "Durée disparition (images)", 8, 1, 120, "gVis");

  defGroup(ps, "gMerge", "Fusion et bords");
  defDouble(ps, "merge", "Fusion des formes", 0.10, 0.0, 2.0, "gMerge",
            "0 = formes nettes et séparées ; plus haut = les formes proches se rejoignent");
  defDouble(ps, "threshold", "Seuil de fusion", 0.10, 0.05, 0.95, "gMerge");
  defDouble(ps, "softness", "Douceur du bord", 0.0, 0.0, 0.5, "gMerge");

  defGroup(ps, "gFx", "Effets");
  defInt(ps, "trail", "Traînée (images)", 0, 0, 30, "gFx");
  defDouble(ps, "smooth", "Lissage des trajectoires", 2.1, 0.0, 5.0, "gFx");
  defBool(ps, "invert", "Inverser la matte", true, "gFx");

  defGroup(ps, "gShadow", "Ombre portée", false);
  defBool(ps, "shadowOn", "Ombre portée", false, "gShadow");
  defDouble(ps, "shadowAngle", "Direction (°)", 135.0, -180.0, 360.0, "gShadow");
  defDouble(ps, "shadowDist", "Distance (px)", 12.0, 0.0, 400.0, "gShadow");
  defDouble(ps, "shadowSoft", "Flou (px)", 6.0, 0.0, 100.0, "gShadow");
  defDouble(ps, "shadowOpacity", "Opacité", 0.6, 0.0, 1.0, "gShadow");
  defBool(ps, "shadowDepth", "Distance selon la profondeur (suivi)", false, "gShadow");

  defGroup(ps, "gEcho", "Écho temporel / slit-scan", false);
  defChoice(ps, "echoMode", "Mode", kEchoLabels, 6, kEchoNone, "gEcho");
  defInt(ps, "echoCount", "Nombre d'échos", 4, 1, 30, "gEcho");
  defInt(ps, "echoStep", "Intervalle (images)", 3, 1, 60, "gEcho");
  defDouble(ps, "echoDecay", "Atténuation", 0.6, 0.0, 1.0, "gEcho");
  defDouble(ps, "echoScale", "Échelle par écho", 1.0, 0.3, 2.0, "gEcho");
  defInt(ps, "slitSpan", "Décalage temporel max. (images)", 24, 1, 240, "gEcho");
}

Params readParams(Instance* in, OfxTime t) {
  Params p;
  auto D = [&](const char* n, double& v) { gParam->paramGetValueAtTime(in->params[n], t, &v); };
  auto I = [&](const char* n, int& v) { gParam->paramGetValueAtTime(in->params[n], t, &v); };
  auto B = [&](const char* n, bool& v) { int x = 0; gParam->paramGetValueAtTime(in->params[n], t, &x); v = x != 0; };
  auto S = [&](const char* n, std::string& v) { char* s = nullptr; gParam->paramGetValueAtTime(in->params[n], t, &s); v = s ? s : ""; };
  S("file", p.file); I("group", p.group); I("offset", p.offset); I("output", p.output);
  B("showPoints", p.showPoints); I("shape", p.shape); S("image", p.image);
  D("size", p.size); D("shapeW", p.shapeW); D("shapeH", p.shapeH); D("opacity", p.opacity); D("rotation", p.rotation); B("follow", p.follow);
  D("jitter", p.jitter); D("grow", p.grow); D("stretch", p.stretch); D("maxScale", p.maxScale);
  B("always", p.always); B("fadeInOn", p.fadeInOn); I("fadeIn", p.fadeIn);
  B("fadeOutOn", p.fadeOutOn); I("fadeOut", p.fadeOut); D("merge", p.merge);
  D("threshold", p.threshold); D("softness", p.softness); I("trail", p.trail);
  D("smooth", p.smooth); B("invert", p.invert);
  B("perspOn", p.perspOn); D("persp", p.persp); I("background", p.background);
  B("shadowOn", p.shadowOn); D("shadowAngle", p.shadowAngle); D("shadowDist", p.shadowDist);
  D("shadowSoft", p.shadowSoft); D("shadowOpacity", p.shadowOpacity); B("shadowDepth", p.shadowDepth);
  I("echoMode", p.echoMode); I("echoCount", p.echoCount); I("echoStep", p.echoStep);
  D("echoDecay", p.echoDecay); D("echoScale", p.echoScale); I("slitSpan", p.slitSpan);
  return p;
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------
OfxStatus onLoad() {
  if (!gHost) return kOfxStatErrMissingHostFeature;
  gProp = static_cast<const OfxPropertySuiteV1*>(gHost->fetchSuite(gHost->host, kOfxPropertySuite, 1));
  gEffect = static_cast<const OfxImageEffectSuiteV1*>(gHost->fetchSuite(gHost->host, kOfxImageEffectSuite, 1));
  gParam = static_cast<const OfxParameterSuiteV1*>(gHost->fetchSuite(gHost->host, kOfxParameterSuite, 1));
  if (!gProp || !gEffect || !gParam) return kOfxStatErrMissingHostFeature;
  static ShapeLib lib;
  gShapes = &lib;
  return kOfxStatOK;
}

OfxStatus describe(OfxImageEffectHandle desc) {
  OfxPropertySetHandle props;
  gEffect->getPropertySet(desc, &props);
  gProp->propSetString(props, kOfxPropLabel, 0, "TAPNext Shapes");
  gProp->propSetString(props, kOfxImageEffectPluginPropGrouping, 0, "TAPNext");
  gProp->propSetString(props, kOfxPropPluginDescription, 0,
                       "Formes et matte alpha sur les points suivis par TAPNext++ (TAPNext Studio).");
  gProp->propSetString(props, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter);
  gProp->propSetString(props, kOfxImageEffectPropSupportedContexts, 1, kOfxImageEffectContextGeneral);
  gProp->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthFloat);
  gProp->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 1, kOfxBitDepthShort);
  gProp->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 2, kOfxBitDepthByte);
  gProp->propSetInt(props, kOfxImageEffectPropSupportsTiles, 0, 0);
  gProp->propSetInt(props, kOfxImageEffectPropSupportsMultiResolution, 0, 1);
  gProp->propSetInt(props, kOfxImageEffectPluginPropSingleInstance, 0, 0);
  gProp->propSetString(props, kOfxImageEffectPluginRenderThreadSafety, 0, kOfxImageEffectRenderFullySafe);
  gProp->propSetInt(props, kOfxImageEffectPluginPropHostFrameThreading, 0, 0);
  gProp->propSetInt(props, kOfxImageEffectPropTemporalClipAccess, 0, 1);  // rush masqué + échos
  return kOfxStatOK;
}

OfxStatus describeInContext(OfxImageEffectHandle desc) {
  OfxPropertySetHandle props;
  gEffect->clipDefine(desc, kOfxImageEffectSimpleSourceClipName, &props);
  gProp->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
  gProp->propSetInt(props, kOfxImageEffectPropTemporalClipAccess, 0, 1);
  gEffect->clipDefine(desc, kOfxImageEffectOutputClipName, &props);
  gProp->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
  describeParams(desc);
  return kOfxStatOK;
}

OfxStatus createInstance(OfxImageEffectHandle effect) {
  auto* in = new Instance();
  in->effect = effect;
  gEffect->clipGetHandle(effect, kOfxImageEffectSimpleSourceClipName, &in->src, nullptr);
  gEffect->clipGetHandle(effect, kOfxImageEffectOutputClipName, &in->dst, nullptr);
  OfxParamSetHandle ps;
  gEffect->getParamSet(effect, &ps);
  for (const char* n : kParamNames) {
    OfxParamHandle h = nullptr;
    gParam->paramGetHandle(ps, n, &h, nullptr);
    in->params[n] = h;
  }
  OfxPropertySetHandle props;
  gEffect->getPropertySet(effect, &props);
  gProp->propSetPointer(props, kOfxPropInstanceData, 0, in);
  // Fichier vide → on mémorise le dernier export de Studio pour ce clip.
  char* cur = nullptr;
  gParam->paramGetValue(in->params["file"], &cur);
  if (!cur || !*cur) {
    std::string last = lastExportPath();
    if (!last.empty()) gParam->paramSetValue(in->params["file"], last.c_str());
  }
  return kOfxStatOK;
}

OfxStatus destroyInstance(OfxImageEffectHandle effect) {
  delete getInstance(effect);
  return kOfxStatOK;
}

OfxStatus getClipPreferences(OfxPropertySetHandle outArgs) {
  gProp->propSetString(outArgs, "OfxImageClipPropComponents_Output", 0, kOfxImageComponentRGBA);
  gProp->propSetString(outArgs, kOfxImageEffectPropPreMultiplication, 0, kOfxImageUnPreMultiplied);
  return kOfxStatOK;
}

template <class T>
inline float toF(T v);
template <> inline float toF<float>(float v) { return v; }
template <> inline float toF<uint16_t>(uint16_t v) { return v / 65535.f; }
template <> inline float toF<uint8_t>(uint8_t v) { return v / 255.f; }
template <class T>
inline T fromF(float v);
template <> inline float fromF<float>(float v) { return v; }
template <> inline uint16_t fromF<uint16_t>(float v) { return uint16_t(std::min(1.f, std::max(0.f, v)) * 65535.f + 0.5f); }
template <> inline uint8_t fromF<uint8_t>(float v) { return uint8_t(std::min(1.f, std::max(0.f, v)) * 255.f + 0.5f); }

struct ImageRef {
  void* data = nullptr;
  int rowBytes = 0;
  OfxRectI b{0, 0, 0, 0};
};

template <class T>
void writePixels(const ImageRef& dst, const ImageRef* src, const OfxRectI& win, const Params& p,
                 const std::vector<float>& matte, int ww, int wh, double fx0, double fy0,
                 double fw, double fh, const std::vector<uint8_t>& dots) {
  for (int y = win.y1; y < win.y2; ++y) {
    T* d = reinterpret_cast<T*>(static_cast<char*>(dst.data) + size_t(y - dst.b.y1) * dst.rowBytes) +
           size_t(win.x1 - dst.b.x1) * 4;
    const T* s = nullptr;
    if (src && src->data && y >= src->b.y1 && y < src->b.y2)
      s = reinterpret_cast<const T*>(static_cast<const char*>(src->data) + size_t(y - src->b.y1) * src->rowBytes);
    // coordonnées normalisées (haut-gauche) → matte de travail
    double yn = 1.0 - ((y + 0.5) - fy0) / fh;
    float my = float(yn * wh - 0.5);
    int iy = int(std::floor(my));
    float ay = my - iy;
    int iy0 = std::max(0, std::min(wh - 1, iy)), iy1 = std::max(0, std::min(wh - 1, iy + 1));
    for (int x = win.x1; x < win.x2; ++x, d += 4) {
      double xn = ((x + 0.5) - fx0) / fw;
      float mx = float(xn * ww - 0.5);
      int ix = int(std::floor(mx));
      float ax = mx - ix;
      int ix0 = std::max(0, std::min(ww - 1, ix)), ix1 = std::max(0, std::min(ww - 1, ix + 1));
      float m = (matte[size_t(iy0) * ww + ix0] * (1 - ax) + matte[size_t(iy0) * ww + ix1] * ax) * (1 - ay) +
                (matte[size_t(iy1) * ww + ix0] * (1 - ax) + matte[size_t(iy1) * ww + ix1] * ax) * ay;
      if (xn < 0 || xn > 1 || yn < 0 || yn > 1) m = p.invert ? 1.f : 0.f;
      float r = 0, g = 0, b = 0, a = 1;
      if (s && x >= src->b.x1 && x < src->b.x2) {
        const T* sp = s + size_t(x - src->b.x1) * 4;
        r = toF(sp[0]); g = toF(sp[1]); b = toF(sp[2]); a = toF(sp[3]);
      }
      if (p.output == 1) { r = g = b = m; a = m; }
      else if (p.output == 2) {
        float k = 0.55f * m;
        r = r * (1 - k) + 1.f * k; g = g * (1 - k) + 0.27f * k; b = b * (1 - k) + 0.24f * k;
      } else { a = m; }
      if (!dots.empty() && dots[size_t(std::max(0, std::min(wh - 1, int(yn * wh)))) * ww +
                                  std::max(0, std::min(ww - 1, int(xn * ww)))]) {
        r = 1.f; g = 0.85f; b = 0.2f; a = 1.f;
      }
      d[0] = fromF<T>(r); d[1] = fromF<T>(g); d[2] = fromF<T>(b); d[3] = fromF<T>(a);
    }
  }
}

ImageRef imageRef(OfxPropertySetHandle img) {
  ImageRef r;
  gProp->propGetPointer(img, kOfxImagePropData, 0, &r.data);
  gProp->propGetInt(img, kOfxImagePropRowBytes, 0, &r.rowBytes);
  gProp->propGetIntN(img, kOfxImagePropBounds, 4, &r.b.x1);
  return r;
}

// ---------------------------------------------------------------------------
// Rush masqué + effets : l'image d'origine dans les formes, échos / slit-scan /
// ombre appliqués à cette image (images voisines lues dans le clip).
// ---------------------------------------------------------------------------
template <class T>
void toBuf(const ImageRef& im, const OfxRectI& win, std::vector<float>& out) {
  const int ww = win.x2 - win.x1, wh = win.y2 - win.y1;
  out.assign(size_t(ww) * wh * 4, 0.f);
  for (int y = std::max(win.y1, im.b.y1); y < std::min(win.y2, im.b.y2); ++y) {
    const T* row = reinterpret_cast<const T*>(static_cast<const char*>(im.data) + size_t(y - im.b.y1) * im.rowBytes);
    float* o = &out[(size_t(y - win.y1) * ww) * 4];
    for (int x = std::max(win.x1, im.b.x1); x < std::min(win.x2, im.b.x2); ++x) {
      const T* sp = row + size_t(x - im.b.x1) * 4;
      float* op = o + size_t(x - win.x1) * 4;
      op[0] = toF(sp[0]); op[1] = toF(sp[1]); op[2] = toF(sp[2]); op[3] = toF(sp[3]);
    }
  }
}

template <class T>
void fromBuf(const ImageRef& dst, const OfxRectI& win, const std::vector<float>& in) {
  const int ww = win.x2 - win.x1;
  for (int y = std::max(win.y1, dst.b.y1); y < std::min(win.y2, dst.b.y2); ++y) {
    T* row = reinterpret_cast<T*>(static_cast<char*>(dst.data) + size_t(y - dst.b.y1) * dst.rowBytes);
    const float* ip = &in[(size_t(y - win.y1) * ww) * 4];
    for (int x = std::max(win.x1, dst.b.x1); x < std::min(win.x2, dst.b.x2); ++x) {
      T* d = row + size_t(x - dst.b.x1) * 4;
      const float* sp = ip + size_t(x - win.x1) * 4;
      d[0] = fromF<T>(sp[0]); d[1] = fromF<T>(sp[1]); d[2] = fromF<T>(sp[2]); d[3] = fromF<T>(sp[3]);
    }
  }
}

bool fetchSrc(OfxImageClipHandle clip, OfxTime tt, const OfxRectI& win, std::vector<float>& out) {
  OfxPropertySetHandle img = nullptr;
  if (gEffect->clipGetImage(clip, tt, nullptr, &img) != kOfxStatOK || !img) return false;
  ImageRef r = imageRef(img);
  char* dep = nullptr;
  gProp->propGetString(img, kOfxImageEffectPropPixelDepth, 0, &dep);
  std::string d = dep ? dep : "";
  if (r.data) {
    if (d == kOfxBitDepthFloat) toBuf<float>(r, win, out);
    else if (d == kOfxBitDepthShort) toBuf<uint16_t>(r, win, out);
    else toBuf<uint8_t>(r, win, out);
  }
  gEffect->clipReleaseImage(img);
  return r.data != nullptr;
}

int framesBackOf(const Params& p) {
  if (p.output != 3) return 0;
  if (p.echoMode == kEcho) return std::min(240, std::max(0, p.echoCount) * std::max(1, p.echoStep));
  if (p.echoMode == kSlitH || p.echoMode == kSlitV || p.echoMode == kSlitRadial) return std::min(240, std::max(1, p.slitSpan));
  return 0;
}

void renderComposite(OfxImageClipHandle srcClip, const TrackFile& tf, const Prepared& pr, const Params& p,
                     const Stamp* stamp, OfxTime time, int t, int ww, int wh, const OfxRectI& win,
                     double fx0, double fy0, double fw, double fh, std::vector<float>& col) {
  const int W = std::max(1, win.x2 - win.x1), H = std::max(1, win.y2 - win.y1);
  const size_t np = size_t(W) * H;
  // fenêtre → matte de travail (bilinéaire)
  std::vector<int> ix0(W), ix1(W), iy0(H), iy1(H);
  std::vector<float> ax(W), ay(H), xn(W), yn(H);
  for (int x = 0; x < W; ++x) {
    xn[x] = float(((win.x1 + x + 0.5) - fx0) / fw);
    float mx = xn[x] * ww - 0.5f;
    int i = int(std::floor(mx));
    ax[x] = mx - i;
    ix0[x] = std::max(0, std::min(ww - 1, i)); ix1[x] = std::max(0, std::min(ww - 1, i + 1));
  }
  for (int y = 0; y < H; ++y) {
    yn[y] = float(1.0 - ((win.y1 + y + 0.5) - fy0) / fh);   // 0 = haut
    float my = yn[y] * wh - 0.5f;
    int i = int(std::floor(my));
    ay[y] = my - i;
    iy0[y] = std::max(0, std::min(wh - 1, i)); iy1[y] = std::max(0, std::min(wh - 1, i + 1));
  }
  auto sampleAll = [&](const std::vector<float>& m, std::vector<float>& out) {
    out.resize(np);
    for (int y = 0; y < H; ++y) {
      const float* r0 = &m[size_t(iy0[y]) * ww];
      const float* r1 = &m[size_t(iy1[y]) * ww];
      for (int x = 0; x < W; ++x) {
        float v = (r0[ix0[x]] * (1 - ax[x]) + r0[ix1[x]] * ax[x]) * (1 - ay[y]) +
                  (r1[ix0[x]] * (1 - ax[x]) + r1[ix1[x]] * ax[x]) * ay[y];
        bool in = xn[x] >= 0 && xn[x] <= 1 && yn[y] >= 0 && yn[y] <= 1;
        out[size_t(y) * W + x] = in ? v : 0.f;
      }
    }
  };
  const float op = float(p.opacity);
  auto base = [&](int tk, float sc, std::vector<float>& m) {
    renderCore(tf, pr, p, stamp, tk, ww, wh, sc, false, m);
    for (auto& v : m) v *= op;
  };
  std::vector<float> cur;
  if (!fetchSrc(srcClip, time, win, cur)) cur.assign(np * 4, 0.f);
  col.assign(np * 4, 0.f);
  if (p.background == 2) col = cur;
  else if (p.background == 3) for (size_t i = 0; i < np * 4; ++i) col[i] = cur[i] * 0.3f;
  std::vector<float> alpha(np, 0.f);
  auto over = [&](const std::vector<float>& src, const std::vector<float>& a) {
    for (size_t i = 0; i < np; ++i) {
      float m = a[i];
      if (m <= 0) continue;
      for (int c = 0; c < 3; ++c) col[i * 4 + c] = col[i * 4 + c] * (1 - m) + src[i * 4 + c] * m;
      alpha[i] = std::max(alpha[i], m);
    }
  };
  std::vector<float> mw, ms;                  // matte de travail / matte fenêtre
  std::vector<float> fxWork;                  // forme qui projette l'ombre
  std::vector<std::pair<std::vector<float>, std::vector<float>>> echoes;
  std::vector<float> mNow, cNow = cur;
  const bool slit = p.echoMode == kSlitH || p.echoMode == kSlitV || p.echoMode == kSlitRadial;
  if (p.echoMode == kEcho) {
    base(t, 1.f, mw);
    fxWork = mw;
    sampleAll(mw, mNow);
    for (int k = std::max(0, p.echoCount); k >= 1; --k) {
      int dk = k * std::max(1, p.echoStep);
      if (t - dk < 0) continue;
      std::vector<float> img;
      if (!fetchSrc(srcClip, time - dk, win, img)) continue;
      std::vector<float> mk;
      base(t - dk, float(std::pow(p.echoScale, k)), mk);
      const float dec = float(std::pow(p.echoDecay, k));
      for (size_t i = 0; i < mk.size(); ++i) { mk[i] *= dec; fxWork[i] = std::max(fxWork[i], mk[i]); }
      std::vector<float> mks;
      sampleAll(mk, mks);
      echoes.emplace_back(std::move(img), std::move(mks));
    }
  } else if (slit) {
    const int span = std::max(1, p.slitSpan);
    const int K = std::min(12, span + 1);
    std::vector<float> U(np);
    const float asp = float(fw / fh), rmax = std::sqrt(0.25f * asp * asp + 0.25f);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        float u = p.echoMode == kSlitH ? xn[x] : p.echoMode == kSlitV ? yn[y]
                : std::sqrt((xn[x] - 0.5f) * (xn[x] - 0.5f) * asp * asp + (yn[y] - 0.5f) * (yn[y] - 0.5f)) / rmax;
        U[size_t(y) * W + x] = std::min(1.f, std::max(0.f, u)) * (K - 1);
      }
    mNow.assign(np, 0.f);
    cNow.assign(np * 4, 0.f);
    for (int i = 0; i < K; ++i) {
      int dk = int(std::lround(double(i) * span / (K - 1)));
      int tk = std::max(0, t - dk);
      std::vector<float> img;
      const std::vector<float>* src = &cur;
      if (dk > 0 && fetchSrc(srcClip, time - dk, win, img)) src = &img;
      base(tk, 1.f, mw);
      sampleAll(mw, ms);
      for (size_t q = 0; q < np; ++q) {
        float w = 1.f - std::fabs(U[q] - i);
        if (w <= 0) continue;
        float m = ms[q] * w;
        mNow[q] += m;
        for (int c = 0; c < 3; ++c) cNow[q * 4 + c] += (*src)[q * 4 + c] * m;
      }
    }
    for (size_t q = 0; q < np; ++q)
      for (int c = 0; c < 3; ++c) cNow[q * 4 + c] = mNow[q] > 1e-6f ? cNow[q * 4 + c] / mNow[q] : 0.f;
    renderTimed(tf, pr, p, stamp, t, ww, wh, fxWork);
    for (auto& v : fxWork) v *= op;
  } else {
    if (p.echoMode == kSlitDepth) renderTimed(tf, pr, p, stamp, t, ww, wh, mw);
    else renderCore(tf, pr, p, stamp, t, ww, wh, 1.f, false, mw);
    for (auto& v : mw) v *= op;
    fxWork = mw;
    sampleAll(mw, mNow);
  }
  if (p.shadowOn && p.shadowOpacity > 0) {
    const float rs = float(ww) / tf.W;
    Canvas sh;
    sh.init(ww, wh);
    float dist = float(p.shadowDist) * rs;
    int dx = int(std::lround(std::cos(p.shadowAngle * kPi / 180) * dist));
    int dy = int(std::lround(std::sin(p.shadowAngle * kPi / 180) * dist));
    for (int y = 0; y < wh; ++y)
      for (int x = 0; x < ww; ++x) {
        int sx = x - dx, sy = y - dy;
        if (sx >= 0 && sx < ww && sy >= 0 && sy < wh) sh.v[size_t(y) * ww + x] = fxWork[size_t(sy) * ww + sx];
      }
    gaussBlur(sh, p.shadowSoft * rs);
    std::vector<float> shs;
    sampleAll(sh.v, shs);
    const float so = float(p.shadowOpacity);
    for (size_t i = 0; i < np; ++i) {
      float a = shs[i] * so;
      for (int c = 0; c < 3; ++c) col[i * 4 + c] *= (1 - a);
      alpha[i] = std::max(alpha[i], a);
    }
  }
  for (auto& e : echoes) over(e.first, e.second);
  over(cNow, mNow);
  for (size_t i = 0; i < np; ++i) {
    if (p.background == 0) {   // non prémultiplié, alpha = couverture
      float a = alpha[i];
      for (int c = 0; c < 3; ++c) col[i * 4 + c] = a > 1e-4f ? col[i * 4 + c] / a : 0.f;
      col[i * 4 + 3] = a;
    } else {
      col[i * 4 + 3] = 1.f;
    }
  }
}

OfxStatus framesNeeded(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
  Instance* in = getInstance(effect);
  if (!in) return kOfxStatReplyDefault;
  OfxTime t = 0;
  gProp->propGetDouble(inArgs, kOfxPropTime, 0, &t);
  Params p = readParams(in, t);
  double range[2] = {t - framesBackOf(p), t};
  gProp->propSetDoubleN(outArgs, "OfxImageClipPropFrameRange_Source", 2, range);
  return kOfxStatOK;
}

OfxStatus render(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs) {
  Instance* in = getInstance(effect);
  if (!in) return kOfxStatFailed;
  OfxTime time = 0;
  gProp->propGetDouble(inArgs, kOfxPropTime, 0, &time);
  OfxRectI win;
  gProp->propGetIntN(inArgs, kOfxImageEffectPropRenderWindow, 4, &win.x1);
  OfxPointD rs{1, 1};
  gProp->propGetDoubleN(inArgs, kOfxImageEffectPropRenderScale, 2, &rs.x);

  OfxPropertySetHandle outImg = nullptr, srcImg = nullptr;
  if (gEffect->clipGetImage(in->dst, time, nullptr, &outImg) != kOfxStatOK) return kOfxStatFailed;
  gEffect->clipGetImage(in->src, time, nullptr, &srcImg);
  ImageRef dst = imageRef(outImg);
  ImageRef src;
  if (srcImg) src = imageRef(srcImg);
  char* depth = nullptr;
  gProp->propGetString(outImg, kOfxImageEffectPropPixelDepth, 0, &depth);

  // Taille de l'image complète (région de définition de la source).
  OfxRectD rod{0, 0, 0, 0};
  gEffect->clipGetRegionOfDefinition(in->src, time, &rod);
  double fx0 = rod.x1 * rs.x, fy0 = rod.y1 * rs.y;
  double fw = std::max(1.0, (rod.x2 - rod.x1) * rs.x), fh = std::max(1.0, (rod.y2 - rod.y1) * rs.y);

  Params p = readParams(in, time);
  if (p.file.empty()) p.file = lastExportPath();

  std::vector<float> matte;
  std::vector<uint8_t> dots;
  int ww = 1, wh = 1;
  {
    std::lock_guard<std::mutex> lock(in->mtx);
    long long stamp = fileStamp(p.file);
    if (p.file != in->loadedPath || stamp != in->loadedStamp) {
      auto tf = std::make_shared<TrackFile>();
      in->tf = loadTrackFile(p.file, *tf) ? tf : nullptr;
      in->loadedPath = p.file;
      in->loadedStamp = stamp;
      in->prepKey.clear();
    }
    if (p.shape == kImage && p.image != in->stampPath) {
      in->stampPath = p.image;
      in->stamp.reset();
      int w, h, c;
      unsigned char* px = p.image.empty() ? nullptr : stbi_load(p.image.c_str(), &w, &h, &c, 4);
      if (px) {
        auto st = std::make_shared<Stamp>();
        st->w = w; st->h = h; st->a.resize(size_t(w) * h);
        bool hasAlpha = false;
        for (int i = 0; i < w * h; ++i) if (px[i * 4 + 3] < 250) { hasAlpha = true; break; }
        for (int i = 0; i < w * h; ++i)
          st->a[i] = hasAlpha ? px[i * 4 + 3] / 255.f
                              : (0.299f * px[i * 4] + 0.587f * px[i * 4 + 1] + 0.114f * px[i * 4 + 2]) / 255.f;
        stbi_image_free(px);
        in->stamp = st;
      }
    }
    if (in->tf) {
      char key[256];
      std::snprintf(key, sizeof(key), "%d|%.4f|%d|%d|%d|%d|%d|%.4f", p.group, p.smooth, p.always,
                    p.fadeInOn, p.fadeIn, p.fadeOutOn, p.fadeOut, p.jitter);
      if (in->prepKey != key || !in->prep) {
        auto pr = std::make_shared<Prepared>();
        prepare(*in->tf, p, *pr);
        in->prep = pr;
        in->prepKey = key;
      }
    }
  }
  std::shared_ptr<TrackFile> tf;
  std::shared_ptr<Prepared> prep;
  std::shared_ptr<Stamp> stampImg;
  {
    std::lock_guard<std::mutex> lock(in->mtx);
    tf = in->tf; prep = in->prep; stampImg = in->stamp;
  }
  if (tf && prep) {
    double sc = std::min(1.0, 1920.0 / std::max(fw, fh));
    ww = std::max(1, int(std::lround(fw * sc)));
    wh = std::max(1, int(std::lround(fh * sc)));
    int t = int(std::lround(time)) + p.offset - tf->first;
    if (p.output == 3) {
      std::vector<float> col;
      if (t >= 0 && t < tf->T) {
        renderComposite(in->src, *tf, *prep, p, stampImg.get(), time, t, ww, wh, win, fx0, fy0, fw, fh, col);
      } else {
        if (!fetchSrc(in->src, time, win, col)) col.assign(size_t(win.x2 - win.x1) * (win.y2 - win.y1) * 4, 0.f);
      }
      std::string dep2 = depth ? depth : "";
      if (dep2 == kOfxBitDepthFloat) fromBuf<float>(dst, win, col);
      else if (dep2 == kOfxBitDepthShort) fromBuf<uint16_t>(dst, win, col);
      else fromBuf<uint8_t>(dst, win, col);
      if (srcImg) gEffect->clipReleaseImage(srcImg);
      gEffect->clipReleaseImage(outImg);
      return kOfxStatOK;
    }
    if (t >= 0 && t < tf->T) {
      renderMatte(*tf, *prep, p, stampImg.get(), t, ww, wh, matte);
      if (p.showPoints) {
        dots.assign(size_t(ww) * wh, 0);
        for (int j = 0; j < prep->n; ++j) {
          size_t i = size_t(t) * prep->n + j;
          if (prep->alpha[i] <= 0.02f || !finite(prep->pos[i * 2])) continue;
          int cx = int(prep->pos[i * 2] * float(ww) / tf->W), cy = int(prep->pos[i * 2 + 1] * float(wh) / tf->H);
          for (int dy = -2; dy <= 2; ++dy)
            for (int dx = -2; dx <= 2; ++dx)
              if (dx * dx + dy * dy <= 4 && cx + dx >= 0 && cx + dx < ww && cy + dy >= 0 && cy + dy < wh)
                dots[size_t(cy + dy) * ww + cx + dx] = 1;
        }
      }
    }
  }
  if (matte.empty()) matte.assign(size_t(ww) * wh, p.invert ? 1.f : 0.f);

  const ImageRef* sp = srcImg ? &src : nullptr;
  std::string dep = depth ? depth : "";
  if (dep == kOfxBitDepthFloat) writePixels<float>(dst, sp, win, p, matte, ww, wh, fx0, fy0, fw, fh, dots);
  else if (dep == kOfxBitDepthShort) writePixels<uint16_t>(dst, sp, win, p, matte, ww, wh, fx0, fy0, fw, fh, dots);
  else writePixels<uint8_t>(dst, sp, win, p, matte, ww, wh, fx0, fy0, fw, fh, dots);

  if (srcImg) gEffect->clipReleaseImage(srcImg);
  gEffect->clipReleaseImage(outImg);
  return kOfxStatOK;
}

OfxStatus mainEntry(const char* action, const void* handle, OfxPropertySetHandle inArgs,
                    OfxPropertySetHandle outArgs) {
  try {
    auto effect = (OfxImageEffectHandle)handle;
    if (!std::strcmp(action, kOfxActionLoad)) return onLoad();
    if (!std::strcmp(action, kOfxActionDescribe)) return describe(effect);
    if (!std::strcmp(action, kOfxImageEffectActionDescribeInContext)) return describeInContext(effect);
    if (!std::strcmp(action, kOfxActionCreateInstance)) return createInstance(effect);
    if (!std::strcmp(action, kOfxActionDestroyInstance)) return destroyInstance(effect);
    if (!std::strcmp(action, kOfxImageEffectActionRender)) return render(effect, inArgs);
    if (!std::strcmp(action, kOfxImageEffectActionGetClipPreferences)) return getClipPreferences(outArgs);
    if (!std::strcmp(action, kOfxImageEffectActionGetFramesNeeded)) return framesNeeded(effect, inArgs, outArgs);
  } catch (...) {
    return kOfxStatFailed;
  }
  return kOfxStatReplyDefault;
}

void setHost(OfxHost* h) { gHost = h; }

OfxPlugin gPlugin = {kOfxImageEffectPluginApi, 1, kPluginId, 1, 0, setHost, mainEntry};

}  // namespace

TAP_EXPORT int OfxGetNumberOfPlugins(void) { return 1; }
TAP_EXPORT OfxPlugin* OfxGetPlugin(int nth) { return nth == 0 ? &gPlugin : nullptr; }
