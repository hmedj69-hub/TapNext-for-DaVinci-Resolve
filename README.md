# TAPNext Studio pour DaVinci Resolve

Tracking de points avec **TAPNext++** de Google DeepMind ([`google-deepmind/tapnet`](https://github.com/google-deepmind/tapnet)) sur des plans 1080p ou 4K de 1024 images et plus. Le résultat sort en **mattes alpha** pour la page Color et en **nœuds Fusion** pour la stabilisation ou le match-move.

```
tap_studio.py         ← TAPNext Studio : application visuelle (recommandé)
shape_engine.py       ← rendu des groupes de formes (mattes)
track_refine.py       ← affinage sous-pixel hybride (TAPNext++ + flux optique)
motion_solver.py      ← solveur de mouvement, stabilisation, export Fusion
ofx_plugin/           ← effet OFX « TAPNext Shapes » pour Resolve (C++, binaire Windows fourni)
tap_resolve_tool.py   ← moteur (tracking, mattes, exports) + ligne de commande
fusion_export.py      ← CSV → nœuds Fusion (sans dépendance ; fonctionne aussi DANS Resolve)
resolve_plugin/       ← script Resolve « Workspace → Scripts → TAPNext_Tracker »
INSTALLER_Windows.bat, install.sh, TAPNext_Studio.bat/.sh, TAPNext_CLI.bat
```

---

## ⚡ Installation en un clic

**Windows** (RTX 3080 Laptop) :

1. Téléchargez le dépôt (bouton **Code → Download ZIP** sur GitHub) et décompressez-le dans un dossier **sans accents**, par exemple `C:\TAPNext`. Évitez aussi les dossiers synchronisés OneDrive.
2. Double-cliquez sur **`INSTALLER_Windows.bat`**. Si SmartScreen s'affiche : *Informations complémentaires → Exécuter quand même*.
   Aucun prérequis : le script installe Python 3.11, PyTorch CUDA, TAPNext++, OpenCV, ffmpeg et Qt, puis télécharge le modèle (~5 Go au total, environ 10 à 20 min). Il ajoute enfin le script dans DaVinci Resolve. **Tout reste dans ce dossier.** Si l'installation est interrompue, relancez le script : il reprend où il s'était arrêté.
3. C'est prêt :
   - **depuis Resolve** : **Workspace → Scripts → TAPNext_Tracker** (redémarrez Resolve s'il était ouvert) ;
   - **ou seul** : double-clic sur **`TAPNext_Studio.bat`** ou sur le raccourci **TAPNext Studio** du Bureau.

**Linux / macOS** : `./install.sh`, puis `./TAPNext_Studio.sh`. Sur macOS, il n'y a pas de CUDA : l'outil tourne sur CPU.

> Pourquoi pas un seul `.exe` ? PyTorch avec CUDA pèse ~3 Go et le modèle 2,5 Go. Un exécutable unique dépasserait 5 Go et serait souvent bloqué par les antivirus. L'installateur produit le même résultat, mais il est réparable et peut être mis à jour.

---

## 🎬 Utilisation depuis DaVinci Resolve

1. Placez la tête de lecture sur le clip à traiter, dans la page Edit ou Color.
2. **Workspace → Scripts → TAPNext_Tracker** : **TAPNext Studio s'ouvre directement sur ce clip**. La plage de suivi est limitée à la partie utilisée dans la timeline.
3. Dans Studio : entourez le sujet, lancez le suivi, réglez la matte. Dans ⑤, choisissez le nœud Fusion (Stabilisation ou Match-move) et l'option « Attacher la matte au clip », puis cliquez sur **Exporter et envoyer à Resolve**.
4. Les résultats arrivent dans Resolve :
   - **automatiquement**, si votre Resolve affiche la petite fenêtre « TAPNext++ » à l'ouverture de Studio ;
   - sinon, il suffit de **relancer Workspace → Scripts → TAPNext_Tracker** une fois l'export terminé (Studio vous le rappelle). C'est le cas notamment sur la version gratuite de Resolve, où les scripts ne peuvent pas ouvrir de fenêtre.
   Une boîte de dialogue confirme ce qui a été importé :
   - la **matte** est attachée au clip. Dans la page **Color** : clic droit dans l'éditeur de nœuds → **Add Matte**, puis reliez la sortie bleue (Key) à l'entrée Key du nœud de correction. Ce câblage reste manuel, car l'API de Resolve ne permet pas de connecter les nœuds de la page Color ;
   - le **nœud Fusion** est ajouté dans la comp du clip, avec des images-clés calées sur la numérotation de la comp. En mode Stabilisation, il est directement inséré avant `MediaOut1`.

**Il ne se passe rien ?**
- Toute erreur s'affiche maintenant dans une boîte de dialogue. Le détail est dans `jobs\resolve_log.txt`, dans le dossier de l'outil, et dans **Workspace → Console**.
- Si le dossier de l'outil a été déplacé, ou si vous avez mis à jour les fichiers, relancez `INSTALLER_Windows.bat`. Il réinstalle le script dans Resolve avec le bon chemin.
- Le dossier de l'outil ne doit pas contenir d'accents (par exemple `C:\TAPNext`).

> Pourquoi pas un « vrai » plugin, effet OFX ou panneau intégré ? Un effet OFX calcule chaque image séparément, et souvent dans le désordre. TAPNext++ doit au contraire lire tout le plan dans l'ordre, sur le GPU, avec PyTorch, ce qu'un effet ne peut pas faire. Les panneaux intégrés (*Workflow Integration Plugins*) sont réservés à DaVinci Resolve Studio. Le script + Studio fonctionne avec toutes les éditions.

---

## 🎯 Stabilisation et tracking (onglet ③ de Studio)

### Comment ça marche
| Étape | Technique | Pourquoi |
|---|---|---|
| Suivi long terme | **TAPNext++** | Suit les points sur des centaines d'images, gère les occultations et **retrouve** les points qui réapparaissent. Aucun décrochage définitif. |
| Précision | **Affinage hybride** : flux optique Lucas-Kanade local sur l'image haute résolution, fusionné avec TAPNext++ par un filtre complémentaire | TAPNext++ prédit dans une grille de 256 px (±4 px en 1080p). Le flux optique apporte le sous-pixel, et TAPNext++ empêche toute dérive. |
| Mouvement | **Solveur RANSAC** : translation, position + rotation + échelle, ou **perspective** (surface plane) | Les points qui bougent autrement (personnages, reflets) sont rejetés. Le calage se fait directement sur l'image de référence, avec relais d'image en image quand les points de référence sortent du champ. |
| Stabilisation | Verrouillée ou lissée, composantes activables, **zoom automatique** | Même logique que le stabilisateur de Resolve ou le Warp Stabilizer. |

Mesuré sur une vidéo de test au mouvement connu (rotation + translation) :

| | TAPNext++ seul | Hybride |
|---|---|---|
| Erreur sur le mouvement de caméra | 1,03 px | **0,33 px** |
| Tremblement après stabilisation | 0,91 px/image | **0,05 px/image** |
| Plan en perspective avec 30 % de points aberrants (modèle Perspective) | — | **0,19 px** |

### Utilisation
1. **① Suivi** : posez une zone de points sur le **décor** pour stabiliser la caméra, ou sur le **sujet ou la surface** à suivre. Laissez « Précision sous-pixel » coché, puis lancez le suivi.
2. **③ Stabiliser** :
   - **Points utilisés** : le groupe qui sert au calcul (par exemple « 1. Zone 1 » pour le décor).
   - **Modèle** : *Position seule*, *Position + rotation + échelle* (caméra à l'épaule, la plupart des cas) ou *Perspective* (surface plane : écran, mur, sol, panneau).
   - **Image de référence** : l'image que la stabilisation verrouille.
   - **Lissage** : 0 = plan verrouillé. Une valeur de 10 à 40 garde le mouvement voulu et retire les tremblements.
   - **Position / Rotation / Échelle** : ce qui est corrigé.
   - **Zoom automatique** : agrandit juste assez pour qu'aucun bord noir n'apparaisse.
   - **Aperçu stabilisé** : la vue montre directement le résultat. Les points doivent rester immobiles.
   - La **frise de temps** colore chaque image selon la qualité du calcul : vert < 0,7 px, jaune < 2 px, rouge au-delà.
3. **Insertion planaire** (modèle Perspective) : glissez les **4 coins orange** sur la surface à remplacer. Ils suivent la surface sur tout le plan.
4. **⑤ Export → Nœuds Fusion** :
   - **Stabilisation** : nœud `Transform`, ou `CornerPositioner` en perspective. Depuis Resolve, il peut être **branché directement** dans la comp du clip.
   - **Match-move** : accroche un élément (texte, logo) au mouvement.
   - **Insertion planaire** : un `CornerPositioner` sur les 4 coins. Branchez votre image d'insertion dessus, puis fusionnez-la sur le plan.

---

## 🧊 Tracker 3D, profondeur, ombres et échos (onglet ④ 3D de Studio)

### Ce que ça calcule
- **Caméra 3D** (équivalent du *Camera Tracker* de Fusion ou de SynthEyes) : à partir des points suivis, Studio retrouve le **mouvement réel de la caméra dans l'espace** et la **position 3D de chaque point**. Technique : couple d'images initial (matrice essentielle), ajout des images par PnP RANSAC, triangulation, puis **ajustement de faisceaux** (bundle adjustment robuste). La **focale est estimée automatiquement** si vous ne la connaissez pas. Si la caméra ne fait que pivoter (panoramique sur pied), Studio le détecte et donne une caméra « rotation seule ».
- **Profondeur de l'image** : une carte de profondeur par image avec l'IA *Depth Anything V2*. Ces réseaux scintillent d'une image à l'autre. Studio les **stabilise avec les points TAPNext++**, car un même point physique doit garder la même profondeur. Si la caméra 3D est résolue, les cartes sont **calées sur la vraie géométrie** des points 3D.
- **Profondeur de chaque point**, à chaque image : c'est ce qu'utilisent les formes (perspective, plage de profondeur, brume, ombres, time-slice).

TAPNext++ convient bien à ce travail : ses pistes sont **longues** et survivent aux occultations, ce qui donne beaucoup d'observations par point (le solveur 3D en a besoin). Le flux optique sous-pixel garde l'erreur de reprojection basse. Sa faiblesse, des points qui glissent sur les objets en mouvement, est filtrée par RANSAC et par l'ajustement robuste.

### Utilisation
1. **① Suivi** : pour la caméra, posez une zone de points sur le **décor fixe**, bien répartie (avant-plan et fond). Pour jouer avec des formes, posez aussi vos groupes sur les sujets.
2. **④ 3D** :
   - **Résoudre la caméra 3D** : choisissez le groupe du décor. Laissez *Focale automatique* cochée, ou entrez l'angle de champ s'il est connu. Les **croix cyan** sont les points 3D reprojetés : elles doivent coller à l'image. L'erreur doit idéalement rester sous 1 px.
   - **Calculer la profondeur** : environ 0,1 s par image sur la RTX 3080. Le modèle (~100 Mo) est téléchargé au premier usage. Vue **« Profondeur »** : rouge = proche, bleu = loin.
3. **② Formes** : chaque groupe a trois nouvelles sections (voir le tableau plus bas). Pour démarrer, essayez les préréglages **Profondeur + ombre**, **Échos fantômes** et **Slit-scan**.
4. **⑤ Export** :
   - **Caméra 3D Fusion** : nœud `TAP_Camera3D` animé, avec ouverture et focale, plus des repères `TAP_Point3D` (Locator3D) sur des points réels de la scène. Depuis Resolve, ils sont collés dans la comp du clip. Reliez la caméra et vos objets 3D à un `Merge3D`, puis à un `Renderer3D` : les objets restent collés à la scène.
   - Fichiers `_points3d.ply` (nuage de points, pour Blender ou d'autres logiciels) et `_camera.json` (caméra par image).
   - **Vidéo de profondeur** `_depth.mov` (blanc = proche), importée dans le chutier TAPNext. Elle sert de matte externe pour isoler un plan de profondeur, ou de carte pour un flou de profondeur.
   - Le fichier `.tapfx` contient la profondeur de chaque point : l'effet OFX dispose des mêmes réglages de profondeur, d'ombre et d'écho.

### Effets créatifs
| Effet | Réglages | Idée |
|---|---|---|
| **Perspective** | Profondeur → Perspective | Les formes rapetissent avec la distance, comme de vrais objets. |
| **Plage de profondeur** | Garder à partir de / jusqu'à, fondu | Masque seulement le premier plan, ou seulement le fond, avec une transition douce. |
| **Brume** | Brume | Les formes lointaines s'effacent. |
| **Ombre portée** | Direction, distance, flou, opacité | Ombre douce derrière le masque. Avec *Distance selon la profondeur*, les formes proches projettent une ombre plus éloignée : effet de relief. |
| **Écho temporel** | Nombre, intervalle, atténuation, échelle | Copies passées du masque qui se superposent et s'estompent : traînées fantômes, effet stroboscopique. |
| **Slit-scan** | Horizontal, vertical ou radial, décalage max. | Chaque bande de l'image montre un instant différent : déformation temporelle façon *2001*. |
| **Time-slice par profondeur** | Décalage max. | Le premier plan est en avance sur le fond : le masque se « déplie » dans la profondeur. |

> Limites : la caméra 3D a besoin d'un **déplacement** de la caméra (parallaxe). Un plan fixe ou un panoramique pur ne donne pas de profondeur géométrique, mais la profondeur IA reste disponible. Les objets qui bougent dans la scène ne doivent pas servir au calcul de la caméra : utilisez un groupe posé sur le décor.

---

## ✨ Effet OFX « TAPNext Shapes » (dans Resolve)

L'installateur ajoute un **vrai effet OpenFX** dans DaVinci Resolve. Toutes les formes se règlent **dans l'Inspecteur de Resolve**, en direct, et l'alpha sort directement sur le nœud de la page Color.

1. Faites le suivi dans **TAPNext Studio**, par exemple depuis **Workspace → Scripts → TAPNext_Tracker**, puis **Exportez**. Studio écrit un fichier de suivi `.tapfx` et le mémorise comme « dernier export ».
2. Dans Resolve, page **Color** : ouvrez la bibliothèque **Effets → OpenFX → TAPNext → TAPNext Shapes** et glissez l'effet sur un nœud (ou sur le clip dans la page Edit).
3. Le champ **Fichier de suivi** se remplit tout seul avec le dernier export. Réglez ensuite dans l'Inspecteur :

| Section | Réglages |
|---|---|
| Suivi TAPNext | Fichier `.tapfx` · **Groupe** (numéro affiché dans la liste de Studio, 0 = tous les points) · Décalage d'image · Sortie · Afficher les points |
| Forme | Forme (cercle, carré, carré arrondi, losange, triangle, hexagone, étoile, croix, anneau, image PNG) · image de forme · taille · opacité · rotation · orienter selon le mouvement · variation de taille |
| Réaction au mouvement | Grossir avec la vitesse · étirer dans la direction · agrandissement maximal |
| Apparition | **Toujours visible** · fondu d'apparition (activable, durée) · fondu de disparition (activable, durée) |
| Fusion et bords | Fusion des formes · seuil · douceur |
| Effets | Traînée · lissage des trajectoires · inverser la matte |
| Profondeur | Taille selon la profondeur · profondeur min./max. · fondu de la plage · brume (nécessite un export avec profondeur) |
| Ombre portée | Ombre · direction · distance · flou · opacité · distance selon la profondeur |
| Écho temporel / slit-scan | Mode (écho, slit-scan horizontal/vertical/radial, time-slice selon la profondeur) · nombre · intervalle · atténuation · échelle · décalage max. |

**Sortie** :
- **Image + alpha** (par défaut) : l'image ne change pas et la matte est dans l'alpha. Dans la page Color, utilisez la sortie Key du nœud, ou mettez l'effet dans un nœud et reliez son alpha à l'entrée Key du nœud de correction.
- **Matte N&B** : la matte en blanc sur noir.
- **Aperçu** : les formes en rouge sur l'image.

Pour donner un style différent à chaque groupe, posez un effet par groupe, chacun avec son numéro de groupe.

Si les formes sont en avance ou en retard sur l'image, ajustez le **Décalage d'image**. L'image 0 du fichier correspond à la première image du média source.

Installation : `INSTALLER_Windows.bat` copie l'effet dans `C:\Program Files\Common Files\OFX\Plugins` (Windows demande une autorisation administrateur). Redémarrez ensuite Resolve. Pour une installation manuelle, copiez le dossier `ofx_plugin\dist\TAPNextShapes.ofx.bundle` à cet endroit. Sous Linux : `sudo ./ofx_plugin/install_ofx.sh`. Sous macOS : compilez avec `ofx_plugin/CMakeLists.txt`.

> L'effet ne fait pas le suivi lui-même. Un effet OFX calcule chaque image séparément, alors que TAPNext++ doit lire tout le plan dans l'ordre, sur le GPU. Le suivi est donc calculé une fois dans Studio, et l'effet dessine les formes à partir de ce suivi. C'est instantané, et chaque réglage se voit tout de suite.

---

## 🖥️ TAPNext Studio, pas à pas

Le panneau de droite a cinq onglets : **① Suivi**, **② Formes**, **③ Stabiliser**, **④ 3D** et **⑤ Export**.

| Étape | Ce que vous faites | Ce qui se passe |
|---|---|---|
| **Placer les points** (①) | Sur une image où le sujet est bien visible, **entourez-le** avec l'outil **Zone** (Z) ou **Rectangle** (R). | La zone se remplit de points (500 par défaut) placés sur les **détails texturés**, ceux que TAPNext++ suit le mieux. **Chaque zone devient un groupe de formes.** L'outil **Point** (A) ajoute un point au groupe sélectionné. **Clic droit** supprime un point. |
| **Suivre** (①) | Réglez Début/Fin (**I**/**O**), puis **Lancer le suivi**. | Suivi **vers l'avant et vers l'arrière**, avec **contrôle aller-retour** : les points qui décrochent sont coupés à l'image exacte du décrochage. Si vous ajoutez des points ensuite, seuls les nouveaux sont suivis. |
| **Formes** (②) | Choisissez un groupe dans la liste et réglez son style. Utilisez un **préréglage** pour démarrer vite. | Le rendu est en direct dans la vue « Image + matte » ou « Matte seule ». Le suivi n'est jamais recalculé. |
| **Éditer à part** (②) | Outil **Sélection** (S) : glissez sur des points (**Maj** pour ajouter), puis **Nouveau groupe avec la sélection**. | Ces points ont désormais leur propre style. Par exemple, des étoiles sur une partie du sujet et des blobs ailleurs. |
| **3D** (④) | Résolvez la caméra 3D et/ou calculez la profondeur. | Les formes peuvent ensuite réagir à la profondeur (voir la section Tracker 3D). |
| **Exporter** (⑤) | Choisissez dossier, format, et éventuellement « une matte par groupe », les nœuds Fusion et la vidéo de contrôle. | La matte est rendue en pleine résolution (1080p ou 4K), et les CSV, JSON et `.setting` sont écrits. |

**Projet** : **Enregistrer le projet** (Ctrl+S) crée un fichier `.tapnext` qui contient les points, le suivi et les formes. Pour le rouvrir : **Ouvrir…**, ou glissez le fichier sur la fenêtre. Rien n'est à recalculer.

### Réglages d'un groupe de formes

| Section | Réglage | Effet |
|---|---|---|
| Forme | Forme | Cercle, carré, carré arrondi, losange, triangle, hexagone, étoile, croix, anneau, ou **votre image PNG** (sa transparence sert de forme). |
| | Révéler / Découper | **Révéler** : la forme est blanche dans la matte. **Découper** : la forme perce un trou dans les autres groupes, ou dans une matte blanche s'il n'y a aucun groupe « Révéler ». |
| | Taille, Opacité, Rotation | Taille en pixels de la vidéo source. |
| | Orienter dans le sens du mouvement | La forme tourne pour suivre la direction du point. |
| | Variation aléatoire de taille | Chaque point reçoit une taille légèrement différente, pour un rendu organique. |
| Réaction au mouvement | Grossir avec la vitesse | La forme grossit quand le point va vite. |
| | Étirer dans la direction | La forme s'allonge dans le sens du mouvement. |
| | Agrandissement maximal | Limite des deux effets précédents. |
| Apparition | **Toujours visible** | La forme reste affichée même quand le point est caché : le masque est toujours actif. |
| | Fondu d'apparition / de disparition | Case à cocher et durée en images. Décochées, la forme apparaît ou disparaît d'un coup. |
| Fusion et bords | Fusion des formes | 0 = formes nettes et séparées. Au-dessus, les formes proches se rejoignent (effet « metaball »). |
| | Seuil, Douceur du bord | Taille de la fusion et adoucissement du contour. |
| Effets | Traînée dans la matte | La forme laisse une traînée qui s'estompe sur N images. |
| | Lissage des trajectoires | Supprime les micro-tremblements. |
| Profondeur | Perspective, plage gardée, fondu, brume | Nécessite l'onglet ④ 3D. 0 = proche, 1 = loin. |
| Ombre portée | Direction, distance, flou, opacité, selon la profondeur | Ombre douce derrière les formes. |
| Écho temporel / slit-scan | Mode, nombre, intervalle, atténuation, échelle, décalage max. | Superpose le masque dans le temps (échos, slit-scan, time-slice). |

Les **valeurs par défaut** reprennent vos réglages : taille 4 px, fusion 0,10, seuil 0,10, douceur 0, grossir 0,1, étirer 0,195, fondus désactivés, lissage 2,1 et matte finale inversée. Le bouton **Style par défaut** enregistre le style courant pour les prochains groupes.

Raccourcis : **Espace** lecture · **←/→** image · **I/O** début/fin · **F** cadrer · **Z** zone · **R** rectangle · **A** point · **S** sélection · **Échap** désélectionner · **Suppr** supprimer · **Ctrl+A** tout sélectionner · **Ctrl+Z** annuler · **Ctrl+O** ouvrir · **Ctrl+S** enregistrer.

Conseils pour un suivi de qualité :
- Posez les points sur une image **nette**, où le sujet est bien visible. Si le sujet change beaucoup d'aspect au cours du plan, ajoutez une seconde zone sur une autre image.
- Gardez **Précision maximale (512 px)** et le **contrôle aller-retour**.
- Entourez **le sujet seul** : des points posés sur le fond suivent le fond.

---

## 1. Installation manuelle (alternative)

Testé avec Python 3.10 et 3.11. Utilisez un environnement virtuel.

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate      Linux/macOS : source .venv/bin/activate

# 1) PyTorch avec CUDA (RTX 3080 Laptop → CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2) Dépendances de l'outil
pip install opencv-python numpy einops tqdm imageio-ffmpeg PySide6-Essentials scipy transformers

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
| | `--verify` | off | Contrôle aller-retour : coupe les pistes qui décrochent (environ 2× plus long) |
| | `--frames-per-step` | 8 (GPU) | Images envoyées ensemble au modèle (même résultat, plus rapide) |
| | `--use-certainty` | off | Visibilité × certitude de position : plus strict, moins de faux « visible » |
| | `--device` | `cuda` | `cuda` ou `cpu` |
| Points | `--grid N` | `10` | Grille N×N (utilisée si aucun point n'est fourni) |
| | `--roi X Y W H` | — | Limite la grille à un rectangle |
| | `--points "x,y;x,y"` / `--points-file` | — | Points fournis à la main |
| | `--normalized` | off | Coordonnées 0..1 au lieu de pixels |
| | `--pick` | off | Sélection à la souris (clic gauche ajoute, clic droit retire, Entrée valide) |
| Temps | `--start-frame` | `0` | Image où les points sont définis |
| | `--end-frame` | `-1` | Dernière image suivie (-1 = fin) |
| | `--backward` | off | Suit aussi de `start-frame` jusqu'à l'image 0 (suivi bidirectionnel) |
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

1. **Tracking** : chaque image est réduite en `input_res × input_res` avec `INTER_AREA` (bon anti-aliasing depuis la 4K) et gardée en mémoire sous cette forme réduite. Elle est ensuite passée à TAPNext++ en mode en ligne : un état récurrent est conservé d'une image à l'autre, les images sont envoyées par paquets de 8 sur GPU, et l'inférence se fait en fp16. Un point posé sur l'image *k* est suivi de *k* vers la fin, puis de *k* vers le début (séquence inversée). Le contrôle aller-retour re-suit chaque piste en sens inverse depuis sa dernière position fiable et localise l'image où les deux sens divergent : la piste y est coupée. Les prédictions sont dans l'espace modèle 256×256 et sont remises à l'échelle en pixels source. La visibilité est la probabilité `sigmoid(visible_logits)` donnée par le modèle.
2. **Post-traitement** : lissage gaussien pondéré par la visibilité, vitesse `v = √(dx² + dy²)` par différence centrée, et enveloppe d'opacité. L'enveloppe suit la visibilité avec une vitesse limitée (`1/fade_in` en montée, `1/fade_out` en descente) : on obtient des fondus réguliers, sans clignotement. Pendant une occultation, le blob reste à la dernière position fiable.
3. **Mattes (metaballs)** : chaque point dessine un cercle ou une ellipse. Le calque est flouté (flou gaussien, σ = `merge × radius`), puis seuillé avec un *smoothstep*. Les blobs proches se rejoignent en un seul masque fluide. L'opacité des fondus vient d'une convolution normalisée, ce qui garde un bord net quand les points sont pleinement visibles. Le champ est calculé en 1080p puis interpolé avant le seuil : les bords restent nets en 4K.
4. **Fusion** : pour chaque image, une similitude 2D (translation, rotation, échelle) est ajustée par moindres carrés pondérés, avec rejet des points aberrants, par rapport à l'image où les points ont été définis. Le résultat est écrit comme keyframes `Center`, `Pivot`, `Angle` et `Size` d'un nœud `Transform`.

### Matériel (RTX 3080 Laptop, 8 Go)

- Poids : ~245 M paramètres, soit ~1 Go en fp32 et ~0,5 Go avec `--fp16-weights`. L'inférence est en fp16 (autocast). En 512 px avec quelques centaines de points, la VRAM reste nettement sous 8 Go. Le pic réel est affiché en fin de tracking.
- En cas d'erreur *CUDA out of memory* : `--points-per-batch 128`, puis `--fp16-weights`, puis `--input-res 256`.
- RAM : les images sont gardées réduites, soit ~0,8 Mo par image en 512 (~800 Mo pour 1024 images). Studio garde en plus un aperçu JPEG (~0,1 Mo par image).
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
