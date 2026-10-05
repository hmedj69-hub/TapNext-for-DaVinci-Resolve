// Mini hôte OpenFX de test (Linux/macOS) pour TAPNext Shapes.
// Charge le plugin, le décrit, crée une instance, règle des paramètres et
// rend quelques images en PGM (alpha) — sans DaVinci Resolve.
//
//   ./mini_host plugin.so fichier.tapfx sortie_prefix W H t1 [t2…] [-- nom=valeur …]
#include <dlfcn.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxParam.h"
#include "ofxProperty.h"

struct PropSet {
  std::map<std::string, std::vector<std::string>> s;
  std::map<std::string, std::vector<double>> d;
  std::map<std::string, std::vector<int>> i;
  std::map<std::string, std::vector<void*>> p;
};
template <class V, class T>
void setv(std::map<std::string, std::vector<V>>& m, const char* k, int idx, T v) {
  auto& vec = m[k];
  if ((int)vec.size() <= idx) vec.resize(idx + 1);
  vec[idx] = v;
}
#define PS(h) reinterpret_cast<PropSet*>(h)

OfxStatus pSetPointer(OfxPropertySetHandle h, const char* k, int i, void* v) { setv(PS(h)->p, k, i, v); return kOfxStatOK; }
OfxStatus pSetString(OfxPropertySetHandle h, const char* k, int i, const char* v) { setv(PS(h)->s, k, i, std::string(v)); return kOfxStatOK; }
OfxStatus pSetDouble(OfxPropertySetHandle h, const char* k, int i, double v) { setv(PS(h)->d, k, i, v); return kOfxStatOK; }
OfxStatus pSetInt(OfxPropertySetHandle h, const char* k, int i, int v) { setv(PS(h)->i, k, i, v); return kOfxStatOK; }
OfxStatus pSetPointerN(OfxPropertySetHandle h, const char* k, int n, void* const* v) { for (int j = 0; j < n; ++j) pSetPointer(h, k, j, v[j]); return kOfxStatOK; }
OfxStatus pSetStringN(OfxPropertySetHandle h, const char* k, int n, const char* const* v) { for (int j = 0; j < n; ++j) pSetString(h, k, j, v[j]); return kOfxStatOK; }
OfxStatus pSetDoubleN(OfxPropertySetHandle h, const char* k, int n, const double* v) { for (int j = 0; j < n; ++j) pSetDouble(h, k, j, v[j]); return kOfxStatOK; }
OfxStatus pSetIntN(OfxPropertySetHandle h, const char* k, int n, const int* v) { for (int j = 0; j < n; ++j) pSetInt(h, k, j, v[j]); return kOfxStatOK; }
OfxStatus pGetPointer(OfxPropertySetHandle h, const char* k, int i, void** v) { auto& m = PS(h)->p; if (!m.count(k) || (int)m[k].size() <= i) return kOfxStatErrUnknown; *v = m[k][i]; return kOfxStatOK; }
OfxStatus pGetString(OfxPropertySetHandle h, const char* k, int i, char** v) { auto& m = PS(h)->s; if (!m.count(k) || (int)m[k].size() <= i) return kOfxStatErrUnknown; *v = (char*)m[k][i].c_str(); return kOfxStatOK; }
OfxStatus pGetDouble(OfxPropertySetHandle h, const char* k, int i, double* v) { auto& m = PS(h)->d; if (!m.count(k) || (int)m[k].size() <= i) return kOfxStatErrUnknown; *v = m[k][i]; return kOfxStatOK; }
OfxStatus pGetInt(OfxPropertySetHandle h, const char* k, int i, int* v) { auto& m = PS(h)->i; if (!m.count(k) || (int)m[k].size() <= i) return kOfxStatErrUnknown; *v = m[k][i]; return kOfxStatOK; }
OfxStatus pGetPointerN(OfxPropertySetHandle h, const char* k, int n, void** v) { for (int j = 0; j < n; ++j) if (pGetPointer(h, k, j, v + j)) return kOfxStatErrUnknown; return kOfxStatOK; }
OfxStatus pGetStringN(OfxPropertySetHandle h, const char* k, int n, char** v) { for (int j = 0; j < n; ++j) if (pGetString(h, k, j, v + j)) return kOfxStatErrUnknown; return kOfxStatOK; }
OfxStatus pGetDoubleN(OfxPropertySetHandle h, const char* k, int n, double* v) { for (int j = 0; j < n; ++j) if (pGetDouble(h, k, j, v + j)) return kOfxStatErrUnknown; return kOfxStatOK; }
OfxStatus pGetIntN(OfxPropertySetHandle h, const char* k, int n, int* v) { for (int j = 0; j < n; ++j) if (pGetInt(h, k, j, v + j)) return kOfxStatErrUnknown; return kOfxStatOK; }
OfxStatus pReset(OfxPropertySetHandle, const char*) { return kOfxStatOK; }
OfxStatus pGetDim(OfxPropertySetHandle h, const char* k, int* c) { auto* s = PS(h); *c = s->s.count(k) ? s->s[k].size() : s->d.count(k) ? s->d[k].size() : s->i.count(k) ? s->i[k].size() : 0; return kOfxStatOK; }
OfxPropertySuiteV1 gPropSuite = {pSetPointer, pSetString, pSetDouble, pSetInt, pSetPointerN, pSetStringN, pSetDoubleN, pSetIntN,
                                 pGetPointer, pGetString, pGetDouble, pGetInt, pGetPointerN, pGetStringN, pGetDoubleN, pGetIntN, pReset, pGetDim};

// --- paramètres
struct Param { std::string type, name; PropSet props; std::string sval; double dval = 0; int ival = 0; double rgb[3] = {0, 0, 0}; };
struct ParamSet { std::map<std::string, std::unique_ptr<Param>> params; std::vector<std::string> order; PropSet props; };
OfxStatus prmDefine(OfxParamSetHandle ps, const char* type, const char* name, OfxPropertySetHandle* props) {
  auto* s = reinterpret_cast<ParamSet*>(ps);
  auto p = std::make_unique<Param>(); p->type = type; p->name = name;
  if (props) *props = (OfxPropertySetHandle)&p->props;
  s->order.push_back(name); s->params[name] = std::move(p);
  return kOfxStatOK;
}
OfxStatus prmGetHandle(OfxParamSetHandle ps, const char* name, OfxParamHandle* h, OfxPropertySetHandle* props) {
  auto* s = reinterpret_cast<ParamSet*>(ps);
  if (!s->params.count(name)) return kOfxStatErrUnknown;
  *h = (OfxParamHandle)s->params[name].get();
  if (props) *props = (OfxPropertySetHandle)&s->params[name]->props;
  return kOfxStatOK;
}
OfxStatus prmSetGetProps(OfxParamSetHandle ps, OfxPropertySetHandle* props) { *props = (OfxPropertySetHandle)&reinterpret_cast<ParamSet*>(ps)->props; return kOfxStatOK; }
OfxStatus prmGetProps(OfxParamHandle h, OfxPropertySetHandle* props) { *props = (OfxPropertySetHandle)&reinterpret_cast<Param*>(h)->props; return kOfxStatOK; }
OfxStatus prmGetValueV(OfxParamHandle h, va_list ap) {
  auto* p = reinterpret_cast<Param*>(h);
  if (p->type == kOfxParamTypeRGB) { for (int k = 0; k < 3; ++k) *va_arg(ap, double*) = p->rgb[k]; }
  else if (p->type == kOfxParamTypeDouble) *va_arg(ap, double*) = p->dval;
  else if (p->type == kOfxParamTypeString) *va_arg(ap, char**) = (char*)p->sval.c_str();
  else *va_arg(ap, int*) = p->ival;
  return kOfxStatOK;
}
OfxStatus prmGetValue(OfxParamHandle h, ...) { va_list ap; va_start(ap, h); auto r = prmGetValueV(h, ap); va_end(ap); return r; }
OfxStatus prmGetValueAtTime(OfxParamHandle h, OfxTime t, ...) { va_list ap; va_start(ap, t); auto r = prmGetValueV(h, ap); va_end(ap); return r; }
OfxStatus prmSetValue(OfxParamHandle h, ...) {
  auto* p = reinterpret_cast<Param*>(h); va_list ap; va_start(ap, h);
  if (p->type == kOfxParamTypeDouble) p->dval = va_arg(ap, double);
  else if (p->type == kOfxParamTypeString) p->sval = va_arg(ap, const char*);
  else p->ival = va_arg(ap, int);
  va_end(ap); return kOfxStatOK;
}
OfxStatus notImpl(...) { return kOfxStatErrUnsupported; }
OfxParameterSuiteV1 gParamSuite;

// --- effet / clips / images
struct Clip { PropSet props; std::string name; };
struct Effect { PropSet props; ParamSet params; std::map<std::string, std::unique_ptr<Clip>> clips; };
int gW = 1920, gH = 1080;
std::vector<float> gSrc, gDst;
std::vector<std::unique_ptr<PropSet>> gImages;
OfxStatus eGetProps(OfxImageEffectHandle e, OfxPropertySetHandle* p) { *p = (OfxPropertySetHandle)&reinterpret_cast<Effect*>(e)->props; return kOfxStatOK; }
OfxStatus eGetParamSet(OfxImageEffectHandle e, OfxParamSetHandle* p) { *p = (OfxParamSetHandle)&reinterpret_cast<Effect*>(e)->params; return kOfxStatOK; }
OfxStatus eClipDefine(OfxImageEffectHandle e, const char* name, OfxPropertySetHandle* p) {
  auto* ef = reinterpret_cast<Effect*>(e); auto c = std::make_unique<Clip>(); c->name = name;
  *p = (OfxPropertySetHandle)&c->props; ef->clips[name] = std::move(c); return kOfxStatOK;
}
OfxStatus eClipGetHandle(OfxImageEffectHandle e, const char* name, OfxImageClipHandle* c, OfxPropertySetHandle* p) {
  auto* ef = reinterpret_cast<Effect*>(e); if (!ef->clips.count(name)) return kOfxStatErrUnknown;
  *c = (OfxImageClipHandle)ef->clips[name].get(); if (p) *p = (OfxPropertySetHandle)&ef->clips[name]->props; return kOfxStatOK;
}
// Source animée (TAP_MOVING=1) : dégradé + barre verticale qui se déplace.
std::map<int, std::vector<float>> gFrames;
float* movingFrame(int t) {
  auto& f = gFrames[t];
  if (f.empty()) {
    f.assign((size_t)gW * gH * 4, 1.f);
    int bx = (t * 15) % gW;
    for (int y = 0; y < gH; ++y)
      for (int x = 0; x < gW; ++x) {
        float* p = &f[((size_t)y * gW + x) * 4];
        p[0] = 0.15f + 0.5f * x / gW; p[1] = 0.15f + 0.5f * y / gH; p[2] = 0.3f;
        if (std::abs(x - bx) < 12) p[0] = p[1] = p[2] = 1.f;
      }
  }
  return f.data();
}
OfxStatus eClipGetImage(OfxImageClipHandle c, OfxTime t, const OfxRectD*, OfxPropertySetHandle* img) {
  auto* clip = reinterpret_cast<Clip*>(c);
  auto ps = std::make_unique<PropSet>();
  bool out = clip->name == kOfxImageEffectOutputClipName;
  if (!out && t < 0) return kOfxStatFailed;
  float* data = out ? gDst.data() : (getenv("TAP_MOVING") ? movingFrame((int)t) : gSrc.data());
  pSetPointer((OfxPropertySetHandle)ps.get(), kOfxImagePropData, 0, data);
  pSetInt((OfxPropertySetHandle)ps.get(), kOfxImagePropRowBytes, 0, gW * 4 * 4);
  int b[4] = {0, 0, gW, gH};
  pSetIntN((OfxPropertySetHandle)ps.get(), kOfxImagePropBounds, 4, b);
  pSetString((OfxPropertySetHandle)ps.get(), kOfxImageEffectPropPixelDepth, 0, kOfxBitDepthFloat);
  *img = (OfxPropertySetHandle)ps.get(); gImages.push_back(std::move(ps)); return kOfxStatOK;
}
OfxStatus eClipRelease(OfxPropertySetHandle) { return kOfxStatOK; }
OfxStatus eClipRoD(OfxImageClipHandle, OfxTime, OfxRectD* r) { *r = {0, 0, (double)gW, (double)gH}; return kOfxStatOK; }
OfxImageEffectSuiteV1 gEffectSuite;

const void* fetchSuite(OfxPropertySetHandle, const char* name, int) {
  if (!strcmp(name, kOfxPropertySuite)) return &gPropSuite;
  if (!strcmp(name, kOfxParameterSuite)) return &gParamSuite;
  if (!strcmp(name, kOfxImageEffectSuite)) return &gEffectSuite;
  return nullptr;
}

int main(int argc, char** argv) {
  if (argc < 7) { fprintf(stderr, "usage\n"); return 1; }
  memset(&gParamSuite, 0, sizeof(gParamSuite));
  gParamSuite.paramDefine = prmDefine; gParamSuite.paramGetHandle = prmGetHandle;
  gParamSuite.paramSetGetPropertySet = prmSetGetProps; gParamSuite.paramGetPropertySet = prmGetProps;
  gParamSuite.paramGetValue = prmGetValue; gParamSuite.paramGetValueAtTime = prmGetValueAtTime;
  gParamSuite.paramSetValue = prmSetValue;
  memset(&gEffectSuite, 0, sizeof(gEffectSuite));
  gEffectSuite.getPropertySet = eGetProps; gEffectSuite.getParamSet = eGetParamSet;
  gEffectSuite.clipDefine = eClipDefine; gEffectSuite.clipGetHandle = eClipGetHandle;
  gEffectSuite.clipGetImage = eClipGetImage; gEffectSuite.clipReleaseImage = eClipRelease;
  gEffectSuite.clipGetRegionOfDefinition = eClipRoD;

  void* lib = dlopen(argv[1], RTLD_NOW);
  if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
  auto nb = (int (*)())dlsym(lib, "OfxGetNumberOfPlugins");
  auto get = (OfxPlugin * (*)(int)) dlsym(lib, "OfxGetPlugin");
  printf("plugins: %d\n", nb());
  OfxPlugin* pl = get(getenv("TAP_PLUGIN") ? atoi(getenv("TAP_PLUGIN")) : 0);
  printf("id: %s v%u.%u api=%s\n", pl->pluginIdentifier, pl->pluginVersionMajor, pl->pluginVersionMinor, pl->pluginApi);
  PropSet hostProps;
  OfxHost host{(OfxPropertySetHandle)&hostProps, fetchSuite};
  pl->setHost(&host);
  printf("load: %d\n", pl->mainEntry(kOfxActionLoad, nullptr, nullptr, nullptr));
  Effect desc;
  printf("describe: %d\n", pl->mainEntry(kOfxActionDescribe, &desc, nullptr, nullptr));
  printf("label: %s / groupe: %s\n", desc.props.s[kOfxPropLabel][0].c_str(), desc.props.s[kOfxImageEffectPluginPropGrouping][0].c_str());
  PropSet ctxArgs; pSetString((OfxPropertySetHandle)&ctxArgs, kOfxImageEffectPropContext, 0, kOfxImageEffectContextFilter);
  printf("describeInContext: %d\n", pl->mainEntry(kOfxImageEffectActionDescribeInContext, &desc, (OfxPropertySetHandle)&ctxArgs, nullptr));
  printf("paramètres (%zu) :", desc.params.order.size());
  for (auto& n : desc.params.order) printf(" %s", n.c_str());
  printf("\n");
  // instance : copie des paramètres avec leurs valeurs par défaut
  gW = atoi(argv[4]); gH = atoi(argv[5]);
  Effect inst;
  for (auto& n : desc.params.order) {
    auto& dp = desc.params.params[n];
    OfxPropertySetHandle pp; prmDefine((OfxParamSetHandle)&inst.params, dp->type.c_str(), n.c_str(), &pp);
    auto& ip = inst.params.params[n];
    ip->props = dp->props;
    if (dp->props.d.count(kOfxParamPropDefault)) ip->dval = dp->props.d[kOfxParamPropDefault][0];
    if (dp->props.i.count(kOfxParamPropDefault)) ip->ival = dp->props.i[kOfxParamPropDefault][0];
    if (dp->type == kOfxParamTypeRGB)
      for (int k = 0; k < 3; ++k) ip->rgb[k] = dp->props.d[kOfxParamPropDefault][k];
    if (dp->props.s.count(kOfxParamPropDefault)) ip->sval = dp->props.s[kOfxParamPropDefault][0];
  }
  for (auto& c : desc.clips) { auto cc = std::make_unique<Clip>(); cc->name = c.first; inst.clips[c.first] = std::move(cc); }
  printf("createInstance: %d\n", pl->mainEntry(kOfxActionCreateInstance, &inst, nullptr, nullptr));
  printf("fichier après création (dernier export) : '%s'\n", inst.params.params["file"]->sval.c_str());
  inst.params.params["file"]->sval = argv[2];
  int ai = 6;
  std::vector<double> times;
  for (; ai < argc && strcmp(argv[ai], "--"); ++ai) times.push_back(atof(argv[ai]));
  for (++ai; ai < argc; ++ai) {
    std::string kv = argv[ai]; auto eq = kv.find('=');
    auto& p = inst.params.params[kv.substr(0, eq)]; std::string v = kv.substr(eq + 1);
    if (p->type == kOfxParamTypeDouble) p->dval = atof(v.c_str());
    else if (p->type == kOfxParamTypeString) p->sval = v;
    else p->ival = atoi(v.c_str());
  }
  gSrc.assign((size_t)gW * gH * 4, 0.5f); gDst.assign((size_t)gW * gH * 4, 0.f);
  for (double t : times) {
    PropSet args;
    pSetDouble((OfxPropertySetHandle)&args, kOfxPropTime, 0, t);
    int win[4] = {0, 0, gW, gH}; pSetIntN((OfxPropertySetHandle)&args, kOfxImageEffectPropRenderWindow, 4, win);
    double rs[2] = {1, 1}; pSetDoubleN((OfxPropertySetHandle)&args, kOfxImageEffectPropRenderScale, 2, rs);
    OfxStatus st = pl->mainEntry(kOfxImageEffectActionRender, &inst, (OfxPropertySetHandle)&args, nullptr);
    char fn[512]; snprintf(fn, sizeof(fn), "%s_%04d.pgm", argv[3], (int)t);
    FILE* f = fopen(fn, "wb"); fprintf(f, "P5\n%d %d\n255\n", gW, gH);
    for (int y = gH - 1; y >= 0; --y)  // OFX : origine en bas → PGM : haut
      for (int x = 0; x < gW; ++x) { float a = gDst[((size_t)y * gW + x) * 4 + 3]; fputc((int)(std::min(1.f, std::max(0.f, a)) * 255 + 0.5f), f); }
    fclose(f);
    if (getenv("TAP_PPM")) {  // sortie couleur (RVB)
      snprintf(fn, sizeof(fn), "%s_%04d.ppm", argv[3], (int)t);
      FILE* g = fopen(fn, "wb"); fprintf(g, "P6\n%d %d\n255\n", gW, gH);
      for (int y = gH - 1; y >= 0; --y)
        for (int x = 0; x < gW; ++x)
          for (int c = 0; c < 3; ++c) fputc((int)(std::min(1.f, std::max(0.f, gDst[((size_t)y * gW + x) * 4 + c])) * 255 + 0.5f), g);
      fclose(g);
    }
    {
      PropSet fa, fo; pSetDouble((OfxPropertySetHandle)&fa, kOfxPropTime, 0, t);
      if (pl->mainEntry(kOfxImageEffectActionGetFramesNeeded, &inst, (OfxPropertySetHandle)&fa, (OfxPropertySetHandle)&fo) == kOfxStatOK)
        printf("frames needed: [%g, %g]\n", fo.d["OfxImageClipPropFrameRange_Source"][0], fo.d["OfxImageClipPropFrameRange_Source"][1]);
    }
    printf("render t=%g : %d → %s\n", t, st, fn);
  }
  pl->mainEntry(kOfxActionDestroyInstance, &inst, nullptr, nullptr);
  return 0;
}
