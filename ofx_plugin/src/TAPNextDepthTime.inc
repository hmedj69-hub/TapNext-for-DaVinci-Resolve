// TAPNext Profondeur & Temps — second effet OpenFX du module (inclus dans
// TAPNextShapes.cpp, même espace de noms anonyme : réutilise les suites OFX,
// les fichiers UTF-8 et les définitions de paramètres).
//
// Agit DIRECTEMENT SUR L'IMAGE du clip :
//   • profondeur (fichier .tapdepth exporté par TAPNext Studio) : isoler une
//     plage de profondeur dans l'alpha (étalonner l'avant-plan ou le fond),
//     brume, flou selon la distance (mise au point virtuelle) ;
//   • temps : écho (traînées), slit-scan horizontal / vertical / radial,
//     time-slice selon la profondeur — à partir des images voisines du clip.
//
// Format .tapdepth (little endian) :
//   char[8] "TAPDEP01" · int32 w, h, T, first_frame, srcW, srcH, reserved[2]
//   float32 lo, hi · uint16 depth[T][h][w] (0 = proche … 65535 = loin)

namespace dt {

const char* kPluginId2 = "com.tapnext.depthtime";
const char* kOut2Labels[] = {"Image traitée (alpha = plage de profondeur)",
                             "Matte de la plage de profondeur (N&B)",
                             "Carte de profondeur (blanc = proche)"};
const char* kTimeLabels[] = {"Aucun", "Écho (traînées)", "Slit-scan horizontal", "Slit-scan vertical",
                             "Slit-scan radial", "Time-slice selon la profondeur"};
const char* kBlendLabels[] = {"Moyenne (fondu)", "Plus clair (max)", "Écran (lumineux)"};
enum { tNone, tEcho, tSlitH, tSlitV, tSlitR, tDepth };
const int kMaxSpan = 240;
const long long kDepthHeader = 8 + 32 + 8;

struct P2 {
  std::string file;
  int offset = 0, output = 0;
  bool rangeOn = false, rangeInv = false;
  double nearV = 0.0, farV = 0.4, feather = 0.05;
  double fog = 0.0, fogStart = 0.3, fogR = 0.75, fogG = 0.8, fogB = 0.85;
  double blur = 0.0, focus = 0.0, focusRange = 0.1;
  int timeMode = tNone, echoCount = 4, echoStep = 2, blend = 0, span = 24;
  double echoDecay = 0.7;
  bool reverse = false, timeInRange = false;
  double mix = 1.0;
};

inline int seek64(FILE* f, long long off) {
#if defined(_WIN32)
  return _fseeki64(f, off, SEEK_SET);
#else
  return fseeko(f, (off_t)off, SEEK_SET);
#endif
}

struct DepthFile {
  int w = 0, h = 0, T = 0, first = 0, srcW = 0, srcH = 0;
  std::string path;
  std::map<int, std::shared_ptr<std::vector<float>>> cache;
  std::vector<int> order;
};

bool openDepth(const std::string& path, DepthFile& d) {
  if (path.empty()) return false;
  FilePtr f(uopen(path, "rb"));
  if (!f) return false;
  char magic[8];
  int32_t hdr[8];
  float lohi[2];
  if (!readAll(f.get(), magic, 8) || std::memcmp(magic, "TAPDEP0", 7) != 0) return false;
  if (!readAll(f.get(), hdr, sizeof(hdr)) || !readAll(f.get(), lohi, sizeof(lohi))) return false;
  d.w = hdr[0]; d.h = hdr[1]; d.T = hdr[2]; d.first = hdr[3]; d.srcW = hdr[4]; d.srcH = hdr[5];
  if (d.w <= 0 || d.h <= 0 || d.T <= 0 || d.w > 8192 || d.h > 8192 || d.T > 10000000) return false;
  d.path = path;
  return true;
}

// Carte de profondeur d'une image (0..1), avec petit cache (appel sous verrou).
std::shared_ptr<std::vector<float>> depthFrame(DepthFile& d, int i) {
  i = std::max(0, std::min(d.T - 1, i));
  auto it = d.cache.find(i);
  if (it != d.cache.end()) return it->second;
  FilePtr f(uopen(d.path, "rb"));
  if (!f) return nullptr;
  const size_t n = size_t(d.w) * d.h;
  if (seek64(f.get(), kDepthHeader + (long long)i * (long long)n * 2) != 0) return nullptr;
  std::vector<uint16_t> buf(n);
  if (!readAll(f.get(), buf.data(), n * 2)) return nullptr;
  auto out = std::make_shared<std::vector<float>>(n);
  for (size_t k = 0; k < n; ++k) (*out)[k] = buf[k] / 65535.f;
  d.cache[i] = out;
  d.order.push_back(i);
  if (d.order.size() > 64) {
    d.cache.erase(d.order.front());
    d.order.erase(d.order.begin());
  }
  return out;
}

struct Instance2 {
  OfxImageEffectHandle effect = nullptr;
  OfxImageClipHandle src = nullptr, dst = nullptr;
  std::map<std::string, OfxParamHandle> params;
  std::mutex mtx;
  std::string loadedPath;
  long long loadedStamp = -2;
  std::shared_ptr<DepthFile> df;
};

Instance2* getInst2(OfxImageEffectHandle effect) {
  OfxPropertySetHandle props;
  gEffect->getPropertySet(effect, &props);
  void* p = nullptr;
  gProp->propGetPointer(props, kOfxPropInstanceData, 0, &p);
  return static_cast<Instance2*>(p);
}

const char* kParamNames2[] = {"file", "offset", "output", "rangeOn", "nearV", "farV", "feather",
                              "rangeInv", "fog", "fogStart", "fogColor", "blur", "focus",
                              "focusRange", "timeMode", "echoCount", "echoStep", "echoDecay",
                              "blend", "span", "reverse", "timeInRange", "mix"};

void describeParams2(OfxImageEffectHandle desc) {
  OfxParamSetHandle ps;
  gEffect->getParamSet(desc, &ps);
  defGroup(ps, "gSrc", "Profondeur (TAPNext Studio)");
  defString(ps, "file", "Fichier de profondeur (.tapdepth)", true, "gSrc",
            "Exporté par TAPNext Studio (onglet ④ 3D). Vide = dernier export.");
  defInt(ps, "offset", "Décalage d'image", 0, -100000, 100000, "gSrc",
         "À ajuster si la profondeur est en avance ou en retard sur l'image");
  defChoice(ps, "output", "Sortie", kOut2Labels, 3, 0, "gSrc");

  defGroup(ps, "gRange", "Isoler une plage de profondeur");
  defBool(ps, "rangeOn", "Activer (résultat dans l'alpha)", false, "gRange",
          "Page Color : la sortie alpha du nœud isole cette plage pour l'étalonner");
  defDouble(ps, "nearV", "Depuis (0 = tout proche)", 0.0, 0.0, 1.0, "gRange");
  defDouble(ps, "farV", "Jusqu'à (1 = horizon)", 0.4, 0.0, 1.0, "gRange");
  defDouble(ps, "feather", "Fondu des bords", 0.05, 0.0, 0.5, "gRange");
  defBool(ps, "rangeInv", "Inverser", false, "gRange");

  defGroup(ps, "gFog", "Brume / atmosphère", false);
  defDouble(ps, "fog", "Intensité", 0.0, 0.0, 1.0, "gFog");
  defDouble(ps, "fogStart", "Commence à (profondeur)", 0.3, 0.0, 1.0, "gFog");
  {
    auto pr = defParam(ps, kOfxParamTypeRGB, "fogColor", "Couleur", nullptr, "gFog");
    gProp->propSetDouble(pr, kOfxParamPropDefault, 0, 0.75);
    gProp->propSetDouble(pr, kOfxParamPropDefault, 1, 0.80);
    gProp->propSetDouble(pr, kOfxParamPropDefault, 2, 0.85);
  }

  defGroup(ps, "gBlur", "Flou de profondeur (mise au point)", false);
  defDouble(ps, "blur", "Flou maximal (px)", 0.0, 0.0, 60.0, "gBlur");
  defDouble(ps, "focus", "Distance nette", 0.0, 0.0, 1.0, "gBlur",
            "0 = premier plan net, 1 = fond net");
  defDouble(ps, "focusRange", "Profondeur de champ", 0.1, 0.0, 1.0, "gBlur");

  defGroup(ps, "gTime", "Temps : écho, slit-scan, time-slice", false);
  defChoice(ps, "timeMode", "Effet", kTimeLabels, 6, tNone, "gTime");
  defInt(ps, "echoCount", "Nombre d'échos", 4, 1, 30, "gTime");
  defInt(ps, "echoStep", "Intervalle (images)", 2, 1, 30, "gTime");
  defDouble(ps, "echoDecay", "Persistance", 0.7, 0.0, 1.0, "gTime");
  defChoice(ps, "blend", "Mélange des échos", kBlendLabels, 3, 0, "gTime");
  defInt(ps, "span", "Décalage temporel max. (images)", 24, 1, kMaxSpan, "gTime",
         "Slit-scan / time-slice : écart de temps entre les deux bords (ou proche / loin)");
  defBool(ps, "reverse", "Inverser le sens", false, "gTime");
  defBool(ps, "timeInRange", "Seulement dans la plage de profondeur", false, "gTime");
  defDouble(ps, "mix", "Mélange avec l'original", 1.0, 0.0, 1.0, nullptr);
}

P2 readParams2(Instance2* in, OfxTime t) {
  P2 p;
  auto D = [&](const char* n, double& v) { gParam->paramGetValueAtTime(in->params[n], t, &v); };
  auto I = [&](const char* n, int& v) { gParam->paramGetValueAtTime(in->params[n], t, &v); };
  auto B = [&](const char* n, bool& v) { int x = 0; gParam->paramGetValueAtTime(in->params[n], t, &x); v = x != 0; };
  char* s = nullptr;
  gParam->paramGetValueAtTime(in->params["file"], t, &s);
  p.file = s ? s : "";
  I("offset", p.offset); I("output", p.output);
  B("rangeOn", p.rangeOn); D("nearV", p.nearV); D("farV", p.farV); D("feather", p.feather);
  B("rangeInv", p.rangeInv);
  D("fog", p.fog); D("fogStart", p.fogStart);
  gParam->paramGetValueAtTime(in->params["fogColor"], t, &p.fogR, &p.fogG, &p.fogB);
  D("blur", p.blur); D("focus", p.focus); D("focusRange", p.focusRange);
  I("timeMode", p.timeMode); I("echoCount", p.echoCount); I("echoStep", p.echoStep);
  D("echoDecay", p.echoDecay); I("blend", p.blend); I("span", p.span);
  B("reverse", p.reverse); B("timeInRange", p.timeInRange); D("mix", p.mix);
  return p;
}

int framesBack(const P2& p) {
  if (p.timeMode == tEcho) return std::min(kMaxSpan, p.echoCount * p.echoStep);
  if (p.timeMode != tNone) return std::min(kMaxSpan, p.span);
  return 0;
}

// --------------------------------------------------------------- pixels
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
      const float* s = ip + size_t(x - win.x1) * 4;
      d[0] = fromF<T>(s[0]); d[1] = fromF<T>(s[1]); d[2] = fromF<T>(s[2]); d[3] = fromF<T>(s[3]);
    }
  }
}

// Image du clip source à l'instant tt, en flottants RGBA sur la fenêtre.
bool fetchFrame(Instance2* in, OfxTime tt, const OfxRectI& win, std::vector<float>& out) {
  OfxPropertySetHandle img = nullptr;
  if (gEffect->clipGetImage(in->src, tt, nullptr, &img) != kOfxStatOK || !img) return false;
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

// Flou gaussien approché (3 passes de moyenne glissante), RGB seulement.
void boxBlurRGB(std::vector<float>& img, int w, int h, double sigma) {
  if (sigma < 0.3 || w < 2 || h < 2) return;
  const int b = std::max(1, int(std::lround((std::sqrt(4.0 * sigma * sigma / 3.0 + 1.0) - 1.0) / 2.0)));
  std::vector<float> line(std::max(w, h) * 3), tmp(std::max(w, h) * 3);
  auto pass = [&](float* base, int n, size_t stride) {
    for (int k = 0; k < n; ++k)
      for (int c = 0; c < 3; ++c) line[k * 3 + c] = base[k * stride + c];
    for (int it = 0; it < 3; ++it) {
      for (int c = 0; c < 3; ++c) {
        double acc = 0;
        for (int k = -b; k <= b; ++k) acc += line[std::max(0, std::min(n - 1, k)) * 3 + c];
        for (int k = 0; k < n; ++k) {
          tmp[k * 3 + c] = float(acc / (2 * b + 1));
          acc += line[std::min(n - 1, k + b + 1) * 3 + c] - line[std::max(0, k - b) * 3 + c];
        }
      }
      std::swap(line, tmp);
    }
    for (int k = 0; k < n; ++k)
      for (int c = 0; c < 3; ++c) base[k * stride + c] = line[k * 3 + c];
  };
  for (int y = 0; y < h; ++y) pass(&img[size_t(y) * w * 4], w, 4);
  for (int x = 0; x < w; ++x) pass(&img[size_t(x) * 4], h, size_t(w) * 4);
}

inline float smooth01(float a, float b, float x) {
  if (b <= a) return x >= a ? 1.f : 0.f;
  float t = std::min(1.f, std::max(0.f, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

float rangeMatte(const P2& p, float d) {
  float fe = float(std::max(1e-3, p.feather));
  float ain = std::min(1.f, std::max(0.f, (d - float(p.nearV - fe)) / fe));
  float aout = std::min(1.f, std::max(0.f, (float(p.farV + fe) - d) / fe));
  float m = ain * aout;
  return p.rangeInv ? 1.f - m : m;
}

// --------------------------------------------------------------- actions
OfxStatus describe2(OfxImageEffectHandle desc) {
  OfxPropertySetHandle props;
  gEffect->getPropertySet(desc, &props);
  gProp->propSetString(props, kOfxPropLabel, 0, "TAPNext Profondeur & Temps");
  gProp->propSetString(props, kOfxImageEffectPluginPropGrouping, 0, "TAPNext");
  gProp->propSetString(props, kOfxPropPluginDescription, 0,
                       "Profondeur (plage isolée, brume, flou) et effets temporels (écho, slit-scan, "
                       "time-slice) appliqués à l'image, à partir de TAPNext Studio.");
  gProp->propSetString(props, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter);
  gProp->propSetString(props, kOfxImageEffectPropSupportedContexts, 1, kOfxImageEffectContextGeneral);
  gProp->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthFloat);
  gProp->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 1, kOfxBitDepthShort);
  gProp->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 2, kOfxBitDepthByte);
  gProp->propSetInt(props, kOfxImageEffectPropSupportsTiles, 0, 0);
  gProp->propSetInt(props, kOfxImageEffectPropSupportsMultiResolution, 0, 1);
  gProp->propSetInt(props, kOfxImageEffectPluginPropSingleInstance, 0, 0);
  gProp->propSetString(props, kOfxImageEffectPluginRenderThreadSafety, 0, kOfxImageEffectRenderInstanceSafe);
  gProp->propSetInt(props, kOfxImageEffectPluginPropHostFrameThreading, 0, 0);
  gProp->propSetInt(props, kOfxImageEffectPropTemporalClipAccess, 0, 1);
  return kOfxStatOK;
}

OfxStatus describeInContext2(OfxImageEffectHandle desc) {
  OfxPropertySetHandle props;
  gEffect->clipDefine(desc, kOfxImageEffectSimpleSourceClipName, &props);
  gProp->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
  gProp->propSetInt(props, kOfxImageEffectPropTemporalClipAccess, 0, 1);
  gEffect->clipDefine(desc, kOfxImageEffectOutputClipName, &props);
  gProp->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
  describeParams2(desc);
  return kOfxStatOK;
}

OfxStatus createInstance2(OfxImageEffectHandle effect) {
  auto* in = new Instance2();
  in->effect = effect;
  gEffect->clipGetHandle(effect, kOfxImageEffectSimpleSourceClipName, &in->src, nullptr);
  gEffect->clipGetHandle(effect, kOfxImageEffectOutputClipName, &in->dst, nullptr);
  OfxParamSetHandle ps;
  gEffect->getParamSet(effect, &ps);
  for (const char* n : kParamNames2) {
    OfxParamHandle h = nullptr;
    gParam->paramGetHandle(ps, n, &h, nullptr);
    in->params[n] = h;
  }
  OfxPropertySetHandle props;
  gEffect->getPropertySet(effect, &props);
  gProp->propSetPointer(props, kOfxPropInstanceData, 0, in);
  char* cur = nullptr;
  gParam->paramGetValue(in->params["file"], &cur);
  if (!cur || !*cur) {
    std::string last = lastPathFrom("last_tapdepth.txt");
    if (!last.empty()) gParam->paramSetValue(in->params["file"], last.c_str());
  }
  return kOfxStatOK;
}

OfxStatus destroyInstance2(OfxImageEffectHandle effect) {
  delete getInst2(effect);
  return kOfxStatOK;
}

OfxStatus framesNeeded2(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
  Instance2* in = getInst2(effect);
  if (!in) return kOfxStatReplyDefault;
  OfxTime t = 0;
  gProp->propGetDouble(inArgs, kOfxPropTime, 0, &t);
  P2 p = readParams2(in, t);
  int back = framesBack(p);
  double range[2] = {p.reverse && p.timeMode == tEcho ? t : t - back,
                     p.reverse && p.timeMode == tEcho ? t + back : t};
  gProp->propSetDoubleN(outArgs, "OfxImageClipPropFrameRange_Source", 2, range);
  return kOfxStatOK;
}

OfxStatus render2(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs) {
  Instance2* in = getInst2(effect);
  if (!in) return kOfxStatFailed;
  OfxTime time = 0;
  gProp->propGetDouble(inArgs, kOfxPropTime, 0, &time);
  OfxRectI win;
  gProp->propGetIntN(inArgs, kOfxImageEffectPropRenderWindow, 4, &win.x1);
  OfxPointD rs{1, 1};
  gProp->propGetDoubleN(inArgs, kOfxImageEffectPropRenderScale, 2, &rs.x);
  OfxPropertySetHandle outImg = nullptr;
  if (gEffect->clipGetImage(in->dst, time, nullptr, &outImg) != kOfxStatOK) return kOfxStatFailed;
  ImageRef dst = imageRef(outImg);
  char* depthStr = nullptr;
  gProp->propGetString(outImg, kOfxImageEffectPropPixelDepth, 0, &depthStr);
  std::string dep = depthStr ? depthStr : "";

  OfxRectD rod{0, 0, 0, 0};
  gEffect->clipGetRegionOfDefinition(in->src, time, &rod);
  const double fx0 = rod.x1 * rs.x, fy0 = rod.y1 * rs.y;
  const double fw = std::max(1.0, (rod.x2 - rod.x1) * rs.x), fh = std::max(1.0, (rod.y2 - rod.y1) * rs.y);
  const int ww = std::max(1, win.x2 - win.x1), wh = std::max(1, win.y2 - win.y1);
  const size_t np = size_t(ww) * wh;

  P2 p = readParams2(in, time);
  if (p.file.empty()) p.file = lastPathFrom("last_tapdepth.txt");

  // --- profondeur de la fenêtre
  std::shared_ptr<std::vector<float>> dmap;
  std::shared_ptr<DepthFile> df;
  {
    std::lock_guard<std::mutex> lock(in->mtx);
    long long st = fileStamp(p.file);
    if (p.file != in->loadedPath || st != in->loadedStamp) {
      auto d = std::make_shared<DepthFile>();
      in->df = openDepth(p.file, *d) ? d : nullptr;
      in->loadedPath = p.file;
      in->loadedStamp = st;
    }
    df = in->df;
    if (df) {
      int di = int(std::lround(time)) + p.offset - df->first;
      if (di >= 0 && di < df->T) dmap = depthFrame(*df, di);
    }
  }
  std::vector<float> D;
  const bool hasD = dmap && df;
  std::vector<float> xn(ww), yn(wh);
  for (int x = 0; x < ww; ++x) xn[x] = float(((win.x1 + x + 0.5) - fx0) / fw);
  for (int y = 0; y < wh; ++y) yn[y] = float(1.0 - ((win.y1 + y + 0.5) - fy0) / fh);  // 0 = haut
  if (hasD) {
    D.resize(np);
    const int w = df->w, h = df->h;
    const float* m = dmap->data();
    for (int y = 0; y < wh; ++y) {
      float v = std::min(float(h - 1), std::max(0.f, yn[y] * h - 0.5f));
      int v0 = int(v), v1 = std::min(h - 1, v0 + 1);
      float fv = v - v0;
      for (int x = 0; x < ww; ++x) {
        float u = std::min(float(w - 1), std::max(0.f, xn[x] * w - 0.5f));
        int u0 = int(u), u1 = std::min(w - 1, u0 + 1);
        float fu = u - u0;
        D[size_t(y) * ww + x] = (m[v0 * w + u0] * (1 - fu) + m[v0 * w + u1] * fu) * (1 - fv) +
                                (m[v1 * w + u0] * (1 - fu) + m[v1 * w + u1] * fu) * fv;
      }
    }
  }

  // --- image courante et effet temporel
  std::vector<float> cur;
  if (!fetchFrame(in, time, win, cur)) cur.assign(np * 4, 0.f);
  std::vector<float> col = cur;
  std::vector<float> img;
  const int mode = (p.timeMode == tDepth && !hasD) ? tNone : p.timeMode;
  if (mode == tEcho) {
    std::vector<float> acc = cur;
    double wsum = 1.0;
    for (int i = 1; i <= p.echoCount; ++i) {
      OfxTime tt = time + (p.reverse ? 1 : -1) * double(i * p.echoStep);
      if (!fetchFrame(in, tt, win, img)) continue;
      float w = float(std::pow(p.echoDecay, double(i)));
      if (p.blend == 0) {
        for (size_t k = 0; k < np * 4; ++k) acc[k] += w * img[k];
        wsum += w;
      } else if (p.blend == 1) {
        for (size_t k = 0; k < np; ++k)
          for (int c = 0; c < 3; ++c) col[k * 4 + c] = std::max(col[k * 4 + c], img[k * 4 + c] * w);
      } else {
        for (size_t k = 0; k < np; ++k)
          for (int c = 0; c < 3; ++c) {
            float a = std::min(1.f, std::max(0.f, col[k * 4 + c])), b = std::min(1.f, std::max(0.f, img[k * 4 + c] * w));
            col[k * 4 + c] = 1.f - (1.f - a) * (1.f - b);
          }
      }
    }
    if (p.blend == 0)
      for (size_t k = 0; k < np * 4; ++k) col[k] = float(acc[k] / wsum);
  } else if (mode != tNone) {
    // u ∈ [0,1] par pixel : 0 = instant présent, 1 = décalage maximal.
    std::vector<float> U(np);
    for (int y = 0; y < wh; ++y)
      for (int x = 0; x < ww; ++x) {
        float u;
        if (mode == tSlitH) u = xn[x];
        else if (mode == tSlitV) u = yn[y];
        else if (mode == tSlitR) {
          float dx = (xn[x] - 0.5f) * float(fw / fh), dy = yn[y] - 0.5f;
          float rmax = std::sqrt(0.25f * float(fw / fh) * float(fw / fh) + 0.25f);
          u = std::sqrt(dx * dx + dy * dy) / rmax;
        } else u = D[size_t(y) * ww + x];
        u = std::min(1.f, std::max(0.f, u));
        U[size_t(y) * ww + x] = p.reverse ? 1.f - u : u;
      }
    const int span = std::max(1, std::min(kMaxSpan, p.span));
    const int K = std::min(span + 1, 33);
    float umin = 1.f, umax = 0.f;
    for (float u : U) { umin = std::min(umin, u); umax = std::max(umax, u); }
    std::fill(col.begin(), col.end(), 0.f);
    std::vector<float> wsum(np, 0.f);
    const int k0 = std::max(0, int(std::floor(umin * (K - 1)))), k1 = std::min(K - 1, int(std::ceil(umax * (K - 1))));
    for (int k = k0; k <= k1; ++k) {
      int off = int(std::lround(double(k) * span / (K - 1)));
      const std::vector<float>* src = &cur;
      if (off > 0 && fetchFrame(in, time - off, win, img)) src = &img;
      for (size_t i = 0; i < np; ++i) {
        float w = 1.f - std::fabs(U[i] * (K - 1) - k);
        if (w <= 0) continue;
        const float* s = &(*src)[i * 4];
        float* o = &col[i * 4];
        o[0] += w * s[0]; o[1] += w * s[1]; o[2] += w * s[2]; o[3] += w * s[3];
        wsum[i] += w;
      }
    }
    for (size_t i = 0; i < np; ++i) {
      float ws = wsum[i] > 1e-6f ? 1.f / wsum[i] : 0.f;
      for (int c = 0; c < 4; ++c) col[i * 4 + c] = wsum[i] > 1e-6f ? col[i * 4 + c] * ws : cur[i * 4 + c];
    }
  }
  if (mode != tNone && p.timeInRange && hasD)
    for (size_t i = 0; i < np; ++i) {
      float m = rangeMatte(p, D[i]);
      for (int c = 0; c < 4; ++c) col[i * 4 + c] = cur[i * 4 + c] + m * (col[i * 4 + c] - cur[i * 4 + c]);
    }

  // --- flou de profondeur (mise au point virtuelle)
  if (hasD && p.blur > 0.05) {
    const double r = p.blur * rs.x;
    const int L = 3;
    std::vector<std::vector<float>> lev(L + 1);
    lev[0] = col;
    for (int j = 1; j <= L; ++j) {
      lev[j] = col;
      boxBlurRGB(lev[j], ww, wh, r * j / double(L) * 0.5);
    }
    const float span = float(std::max(0.05, 0.5 * (1.0 - p.focusRange)));
    for (size_t i = 0; i < np; ++i) {
      float coc = std::min(1.f, std::max(0.f, (std::fabs(D[i] - float(p.focus)) - float(p.focusRange)) / span));
      float c = coc * L;
      int j = std::min(L - 1, int(c));
      float f = c - j;
      for (int ch = 0; ch < 3; ++ch)
        col[i * 4 + ch] = lev[j][i * 4 + ch] * (1 - f) + lev[j + 1][i * 4 + ch] * f;
    }
  }
  // --- brume
  if (hasD && p.fog > 0) {
    const float fc[3] = {float(p.fogR), float(p.fogG), float(p.fogB)};
    for (size_t i = 0; i < np; ++i) {
      float f = float(p.fog) * smooth01(float(p.fogStart), 1.f, D[i]);
      for (int c = 0; c < 3; ++c) col[i * 4 + c] = col[i * 4 + c] * (1 - f) + fc[c] * f;
    }
  }
  // --- mélange et sortie
  const float mix = float(std::min(1.0, std::max(0.0, p.mix)));
  for (size_t i = 0; i < np; ++i) {
    float* o = &col[i * 4];
    const float* c0 = &cur[i * 4];
    for (int c = 0; c < 3; ++c) o[c] = c0[c] + mix * (o[c] - c0[c]);
    float m = (p.rangeOn && hasD) ? rangeMatte(p, D[i]) : 1.f;
    if (p.output == 1) {
      float v = hasD ? rangeMatte(p, D[i]) : 0.f;
      o[0] = o[1] = o[2] = o[3] = v;
    } else if (p.output == 2) {
      float v = hasD ? 1.f - D[i] : 0.f;
      o[0] = o[1] = o[2] = v;
      o[3] = 1.f;
    } else {
      o[3] = (p.rangeOn && hasD) ? m : c0[3];
    }
  }
  if (dep == kOfxBitDepthFloat) fromBuf<float>(dst, win, col);
  else if (dep == kOfxBitDepthShort) fromBuf<uint16_t>(dst, win, col);
  else fromBuf<uint8_t>(dst, win, col);
  gEffect->clipReleaseImage(outImg);
  return kOfxStatOK;
}

OfxStatus mainEntry2(const char* action, const void* handle, OfxPropertySetHandle inArgs,
                     OfxPropertySetHandle outArgs) {
  try {
    auto effect = (OfxImageEffectHandle)handle;
    if (!std::strcmp(action, kOfxActionLoad)) return onLoad();
    if (!std::strcmp(action, kOfxActionDescribe)) return describe2(effect);
    if (!std::strcmp(action, kOfxImageEffectActionDescribeInContext)) return describeInContext2(effect);
    if (!std::strcmp(action, kOfxActionCreateInstance)) return createInstance2(effect);
    if (!std::strcmp(action, kOfxActionDestroyInstance)) return destroyInstance2(effect);
    if (!std::strcmp(action, kOfxImageEffectActionRender)) return render2(effect, inArgs);
    if (!std::strcmp(action, kOfxImageEffectActionGetFramesNeeded)) return framesNeeded2(effect, inArgs, outArgs);
    if (!std::strcmp(action, kOfxImageEffectActionGetClipPreferences)) return getClipPreferences(outArgs);
  } catch (...) {
    return kOfxStatFailed;
  }
  return kOfxStatReplyDefault;
}

OfxPlugin gPlugin2 = {kOfxImageEffectPluginApi, 1, kPluginId2, 1, 0, setHost, mainEntry2};

}  // namespace dt
