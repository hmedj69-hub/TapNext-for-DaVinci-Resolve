# TAPNext++ pour DaVinci Resolve

Outil Python autonome qui suit des points dans une vidéo (MP4/MOV, 1080p ou 4K,
jusqu'à 1024 images et plus) avec **TAPNext++** de Google DeepMind
([`google-deepmind/tapnet`](https://github.com/google-deepmind/tapnet)), puis produit :

| Sortie | Fichier | Utilisation dans Resolve |
|---|---|---|
| Matte alpha N&B (à la résolution source) | `<clip>_matte.mov` (ProRes 422 HQ) | Page **Color** → *External Matte* |
| Trajectoires | `<clip>_tracks.csv` / `<clip>_tracks.json` | Analyse, scripts, autres logiciels |
| Nœud Transform de stabilisation | `<clip>_fusion_stabilize.setting` | Page **Fusion** (glisser-déposer ou Ctrl+V) |
| Nœud Transform de match-move | `<clip>_fusion_matchmove.setting` | Page **Fusion**, pour accrocher un élément |
| Vidéo de contrôle (option `--preview`) | `<clip>_preview.mp4` | Vérification visuelle |

```
tap_resolve_tool.py   ← script principal (tracking, mattes, exports)
fusion_export.py      ← CSV → nœuds Fusion (sans dépendance ; fonctionne aussi DANS Resolve)
requirements.txt
```

---

## 1. Installation

Testé avec Python 3.10 et 3.11. Utilisez un environnement virtuel.

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate      Linux/macOS : source .venv/bin/activate

# 1) PyTorch avec CUDA (RTX 3080 Laptop → CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2) Dépendances de l'outil
pip install opencv-python numpy einops tqdm imageio-ffmpeg

# 3) TAPNext++ (dépôt DeepMind). --no-deps évite d'installer JAX/Haiku/TensorFlow,
#    inutiles pour l'inférence PyTorch.
pip install --no-deps "tapnet @ git+https://github.com/google-deepmind/tapnet.git"
```

Ou, en une fois après l'étape 1 : `pip install -r requirements.txt`.

- `tapnet` n'est **pas** publié sur PyPI : `pip install tapnet` échouera, il faut l'installer depuis GitHub comme ci-dessus.
- **ffmpeg** sert à encoder la matte en ProRes, DNxHR ou H.264. `imageio-ffmpeg` en fournit un. Sinon, installez ffmpeg et mettez-le dans le PATH. Sans ffmpeg, l'outil exporte une séquence PNG.
- **Checkpoint** : il est téléchargé automatiquement dans `checkpoints/` au premier lancement (~2,5 Go). Vous pouvez aussi le télécharger vous-même :
  ```bash
  # entrée 512 px (défaut, plus précis)
  curl -L -o checkpoints/tapnextpp_512.ckpt https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt
  # entrée 256 px (plus rapide)
  curl -L -o checkpoints/tapnextpp_ckpt.pt  https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt
  ```
- Pour sélectionner les points à la souris (`--pick`), il faut `opencv-python` et non la variante `-headless`.

Vérification :
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python tap_resolve_tool.py --help
```

---

## 2. Utilisation rapide

```bash
# Grille automatique 10×10 sur toute l'image. Produit la matte, le CSV et les nœuds Fusion.
python tap_resolve_tool.py "D:/Rushes/A001_C003.mov"

# Grille 8×8 limitée à un sujet (rectangle x y largeur hauteur, en pixels source)
python tap_resolve_tool.py clip.mov --grid 8 --roi 1400 300 900 1400

# Points précis (pixels de la vidéo source) et blobs étirés selon le mouvement
python tap_resolve_tool.py clip.mp4 --points "960,540;1010,560;900,600" \
    --radius 70 --motion-mode stretch --motion-sensitivity 0.08

# Points cliqués sur l'image 120, suivis en avant ET en arrière, avec vidéo de contrôle
python tap_resolve_tool.py clip.mov --pick --start-frame 120 --backward --preview

# Points lus dans un fichier (JSON [[x,y],…] ou CSV x,y), coordonnées normalisées 0..1
python tap_resolve_tool.py clip.mov --points-file pts.json --normalized

# Matte seule, en DNxHR, sans nœuds Fusion
python tap_resolve_tool.py clip.mov --codec dnxhr --fusion-mode none
```

### Paramètres

Les valeurs par défaut sont regroupées dans le dictionnaire `DEFAULTS`, en haut de
`tap_resolve_tool.py`. Vous pouvez les modifier là, ou les passer en ligne de commande.

| Groupe | Argument | Défaut | Rôle |
|---|---|---|---|
| E/S | `video` | — | Vidéo source MP4/MOV |
| | `-o / --output-dir`, `--name` | dossier de la vidéo | Dossier et préfixe des fichiers de sortie |
| Modèle | `--input-res` | `512` | Résolution interne (256 ou 512). Les images sont réduites, puis les coordonnées sont ramenées à la résolution source |
| | `--checkpoint` | auto | Chemin du `.pt`/`.ckpt` |
| | `--fp16-weights` | off | Poids en float16 : ~0,5 Go de VRAM au lieu de ~1 Go |
| | `--points-per-batch` | `512` | Nombre de points traités ensemble. À baisser si la VRAM manque |
| | `--use-certainty` | off | Visibilité × certitude de position : plus strict, moins de faux « visible » |
| | `--device` | `cuda` | `cuda` ou `cpu` |
| Points | `--grid N` | `10` | Grille N×N (utilisée si aucun point n'est fourni) |
| | `--roi X Y W H` | — | Limite la grille à un rectangle |
| | `--points "x,y;x,y"` / `--points-file` | — | Points fournis à la main |
| | `--normalized` | off | Coordonnées 0..1 au lieu de pixels |
| | `--pick` | off | Sélection à la souris (clic gauche ajoute, clic droit retire, Entrée valide) |
| Temps | `--start-frame` | `0` | Image où les points sont définis |
| | `--end-frame` | `-1` | Dernière image suivie (-1 = fin) |
| | `--backward` | off | Suit aussi de `start-frame` jusqu'à l'image 0 |
| Post-traitement | `--smooth-sigma` | `1.0` | Lissage temporel des trajectoires (en images), pondéré par la visibilité |
| | `--vis-threshold` | `0.5` | Seuil de visibilité |
| | `--fade-in` / `--fade-out` | `6` / `8` | Durée des fondus d'apparition et de disparition (images) |
| Matte | `--radius` | `40` | Rayon de base des blobs (pixels source). Prévoir ~2× en 4K |
| | `--merge` | `0.6` | Flou de fusion, en multiple du rayon. Plus haut = blobs plus fusionnés |
| | `--threshold` | `0.5` | Seuil des metaballs. Plus bas = masques plus gros et plus liés |
| | `--edge-softness` | `0.08` | Douceur du bord (0 = bord dur) |
| | `--motion-mode` | `none` | `scale` (grossit), `stretch` (s'étire dans la direction du mouvement), `both` |
| | `--motion-sensitivity` | `0.05` | Facteur = 1 + gain × v, avec v = √(dx²+dy²) en px/image ramenés à 1080p |
| | `--max-motion-scale` | `3.0` | Facteur maximal |
| | `--render-max-side` | `1920` | Résolution de calcul du champ. Le seuil final est appliqué en pleine résolution |
| | `--codec` | `prores` | `prores` (422 HQ .mov), `dnxhr` (HQ .mov), `h264` (.mp4), `png` |
| | `--invert`, `--preview`, `--no-matte` | off | Inverser la matte, vidéo de contrôle, ne pas générer la matte |
| Fusion | `--fusion-mode` | `both` | `stabilize`, `matchmove`, `both`, `none` |
| | `--fusion-model` | `similarity` | `similarity` (position + rotation + échelle) ou `translation` |
| | `--fusion-smooth` | `0` | 0 = plan verrouillé. N > 0 = stabilisation douce (seules les vibrations sont retirées) |
| | `--fusion-frame-offset` | `0` | Décalage des keyframes (ex. `1001`) |
| | `--fusion-point-paths` | — | IDs de points exportés chacun dans un Transform dédié |

### Fonctionnement

1. **Tracking** : la vidéo est lue en flux (aucune image 4K n'est gardée en mémoire). Chaque image est réduite en `input_res × input_res` avec `INTER_AREA` (bon anti-aliasing depuis la 4K), puis passée à TAPNext++ en mode en ligne : un état récurrent est conservé d'une image à l'autre, et l'inférence se fait en fp16. Les prédictions sont dans l'espace modèle 256×256 et sont remises à l'échelle en pixels source. La visibilité est la probabilité `sigmoid(visible_logits)` donnée par le modèle.
2. **Post-traitement** : lissage gaussien pondéré par la visibilité, vitesse `v = √(dx² + dy²)` par différence centrée, et enveloppe d'opacité. L'enveloppe suit la visibilité avec une vitesse limitée (`1/fade_in` en montée, `1/fade_out` en descente) : on obtient des fondus réguliers, sans clignotement. Pendant une occultation, le blob reste à la dernière position fiable.
3. **Mattes (metaballs)** : chaque point dessine un cercle ou une ellipse. Le calque est flouté (flou gaussien, σ = `merge × radius`), puis seuillé avec un *smoothstep*. Les blobs proches se rejoignent en un seul masque fluide. L'opacité des fondus vient d'une convolution normalisée, ce qui garde un bord net quand les points sont pleinement visibles. Le champ est calculé en 1080p puis interpolé avant le seuil : les bords restent nets en 4K.
4. **Fusion** : pour chaque image, une similitude 2D (translation, rotation, échelle) est ajustée par moindres carrés pondérés, avec rejet des points aberrants, par rapport à l'image où les points ont été définis. Le résultat est écrit comme keyframes `Center`, `Pivot`, `Angle` et `Size` d'un nœud `Transform`.

### Matériel (RTX 3080 Laptop, 8 Go)

- Poids : ~245 M paramètres, soit ~1 Go en fp32 et ~0,5 Go avec `--fp16-weights`. L'inférence est en fp16 (autocast). En 512 px avec quelques centaines de points, la VRAM reste nettement sous 8 Go. Le pic réel est affiché en fin de tracking.
- En cas d'erreur *CUDA out of memory* : `--points-per-batch 128`, puis `--fp16-weights`, puis `--input-res 256`.
- RAM : seul `--backward` met des images en cache, et uniquement en version réduite (~0,8 Mo par image en 512).
- Le modèle a été affiné sur des séquences de 1024 images. Il continue de fonctionner au-delà, mais pour des plans très longs il vaut mieux découper en segments.

### Format du CSV

```
frame_index,point_id,x_pixels,y_pixels,visibility,velocity
0,0,96.000,54.000,0.9998,0.000
0,1,288.000,54.000,0.9995,0.000
...
```

`frame_index` est l'index d'image du fichier source, en commençant à 0. Les coordonnées sont en pixels source, origine en haut à gauche, Y vers le bas. `visibility` est la probabilité de visibilité (0..1). `velocity` est en pixels par image. Le JSON compagnon contient les métadonnées (résolution, fps, timecode, image de référence) et, avec `--json-full`, toutes les trajectoires.

---

## 3. Guide : utiliser la matte dans la page Color

1. **Importer** : glissez `<clip>_matte.mov` dans le **Media Pool**, à côté du clip source.
2. **Associer la matte au clip** : dans le Media Pool, clic droit sur le **clip source** → *Add Matte…* (*Ajouter un cache*), puis choisissez `<clip>_matte.mov`. La matte est alors liée au clip, sur toute sa durée (*clip matte*).
   - Autre méthode : glissez la matte directement dans la page Color, sur l'éditeur de nœuds. Ou ajoutez-la comme *Timeline Matte* (clic droit → *Add as Timeline Matte*). Dans ce cas, la matte doit être calée sur le même timecode : l'outil copie le timecode source dans le `.mov` quand ffprobe est disponible.
3. **Page Color** : dans l'éditeur de nœuds, clic droit sur le fond → **Add Matte** → choisissez la matte. Un nœud *External Matte* (source verte) apparaît.
4. **Connecter** : reliez la sortie **bleue (Key)** du nœud matte à l'**entrée Key** (triangle bleu) du nœud de correction voulu.
5. **Régler** dans la palette **Key** :
   - *Key Input Gain* / *Offset* : intensité de la matte ;
   - *Invert* : pour corriger l'extérieur du masque ;
   - le mode composite et *Blur* permettent d'adoucir encore les bords.
6. **Vérifier** : activez **Highlight** (Shift+H) pour voir la zone affectée.

Notes :
- La matte fait exactement la même résolution et le même nombre d'images que le clip source. Si les points sont définis à partir de `--start-frame`, les images précédentes sont noires.
- Un clip retaillé (*trimmed*) dans la timeline reste synchrone, car la matte est liée au clip source.
- Pour une matte en 4K avec une timeline en 1080p, Resolve applique le même *Input Scaling* que pour le clip.

---

## 4. Guide : utiliser le tracking dans la page Fusion

### Méthode 1 : fichier `.setting` généré

1. Ouvrez le clip dans la page **Fusion**.
2. Glissez `<clip>_fusion_stabilize.setting` dans le **Node Editor**. Vous pouvez aussi ouvrir le fichier dans un éditeur de texte, tout copier, puis faire Ctrl+V dans le Node Editor.
3. Branchez `MediaIn1 → TAP_Stabilize → MediaOut1`.
   - **Stabilize** : `Pivot` suit la caméra, `Center` reste fixe (ou suit une version lissée avec `--fusion-smooth N`), et `Angle`/`Size` compensent rotation et zoom. Ajoutez un `Transform` ou un `Crop` après pour recadrer les bords.
   - **Match-move** (`<clip>_fusion_matchmove.setting`) : branchez l'élément à incruster (Text+, logo…) sur `TAP_MatchMove`, puis fusionnez le résultat avec un `Merge` au-dessus de `MediaIn1`. Placez l'élément tel qu'il doit apparaître sur l'image de référence.
   - `--fusion-point-paths 3,17` crée un Transform `TAP_Point_3`, `TAP_Point_17`… par point. Cela sert à suivre un point précis, par exemple un œil ou un coin d'écran.
4. **Numérotation des images** : l'image 0 du CSV correspond à la première image du fichier source. Si votre comp démarre à un autre numéro, par exemple 1001, ou si le clip est retaillé, utilisez `--fusion-frame-offset`.

### Méthode 2 : script dans Resolve

`fusion_export.py` peut aussi tourner directement dans Resolve, sans aucune dépendance :

1. Copiez `fusion_export.py` dans le dossier des scripts Fusion :
   - Windows : `%APPDATA%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\Comp\`
   - macOS : `~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Comp/`
   - Linux : `~/.local/share/DaVinciResolve/Fusion/Scripts/Comp/`
2. Dans la page Fusion : **Workspace → Scripts → Comp → fusion_export**.
3. Une boîte de dialogue demande le CSV, le mode (Stabilize ou Match-move), le modèle, le lissage, le décalage d'image et les IDs de points. Les nœuds sont alors collés dans la comp.

Le script s'utilise aussi en ligne de commande pour régénérer un `.setting` sans relancer le tracking :

```bash
python fusion_export.py clip_tracks.csv --mode stabilize --smooth 12 -o stab_douce.setting
python fusion_export.py clip_tracks.csv --mode matchmove --points 4,5,10,11 --model translation
```

Conventions : Fusion utilise des coordonnées normalisées, avec l'origine en bas à gauche, soit `xn = x/W` et `yn = 1 − y/H`. Les angles sont en degrés, sens anti-horaire positif.

---

## Licence

Le code de ce dépôt peut être utilisé librement. TAPNext++ (code et checkpoint) est distribué par Google DeepMind sous licence Apache 2.0.
