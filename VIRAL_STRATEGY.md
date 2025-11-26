# Stratégie de Viralité - rgbd-depth

## 🎯 Objectif
Atteindre 1000+ stars GitHub et 10k+ downloads PyPI en 3 mois

## 📊 Plan d'action prioritaire

### Phase 1: Démos visuelles (Semaine 1-2)

#### A. Créer des GIFs/vidéos avant/après
- [ ] GIF comparant depth brute vs refinée
- [ ] Vidéo side-by-side de 3-5 scènes différentes
- [ ] Comparaison vitesse: CPU → MPS → CUDA (+xFormers)
- [ ] Showcase de cas d'usage: robotique, AR, 3D reconstruction

**Outils:** ffmpeg pour GIFs, OBS pour screen capture

#### B. README avec démo interactive
```markdown
## ✨ See it in action

<p align="center">
  <img src="assets/demo.gif" width="800"/>
</p>

### Before vs After
| Input Depth | RGB-D Refined | Improvement |
|-------------|---------------|-------------|
| ![](assets/before1.jpg) | ![](assets/after1.jpg) | **3.2x sharper** |
```

#### C. Google Colab notebook
- [ ] Créer notebook Colab "Try rgbd-depth in 60 seconds"
- [ ] Badge "Open in Colab" dans README
- [ ] Exemples préchargés (pas besoin de télécharger modèle)

### Phase 2: Distribution virale (Semaine 2-3)

#### A. Reddit posts
**Communautés cibles:**
- r/MachineLearning (lundi matin, titre: "I optimized ByteDance's RGB-D depth model - 8% faster with xFormers")
- r/computervision (mercredi)
- r/learnmachinelearning (vendredi)
- r/python (samedi, focus sur PyPI package)

**Format post:**
```
Title: [P] rgbd-depth: Production-ready RGB-D depth refinement (8% faster, PyPI, MPS support)

I've packaged and optimized ByteDance's camera-depth-models research:

✅ pip install rgbd-depth (one command!)
✅ 8% faster with xFormers on CUDA
✅ Apple Silicon (MPS) support - fixed blurry rendering bug
✅ Mixed precision FP16/BF16
✅ Pixel-perfect vs reference implementation

[GIF showing before/after]

GitHub: https://github.com/Aedelon/camera-depth-models
PyPI: https://pypi.org/project/rgbd-depth/

Try it in Colab: [badge]
```

#### B. Hacker News
- [ ] Post un mardi/mercredi 8-10am PT
- [ ] Titre: "Show HN: rgbd-depth – Production RGB-D depth refinement (PyPI package)"
- [ ] Première ligne du post = démo visuelle + lien Colab

#### C. Twitter/X thread
```
🚀 Just released rgbd-depth v1.0.2 on PyPI!

Production-ready RGB-D depth refinement from @ByteDanceLab research

✅ One-line install: pip install rgbd-depth
✅ 8% faster (xFormers)
✅ Apple Silicon support
✅ FP16/BF16 mixed precision

[GIF 1/4]

Thread 👇
```

- 4-5 tweets avec GIFs/screenshots
- Tag: @PyPI, @pytorch, communautés CV/robotics
- Hashtags: #ComputerVision #PyTorch #MachineLearning #Robotics

#### D. LinkedIn post
- Version professionnelle du Twitter thread
- Focus: "How I turned research code into production package"
- Tag entreprises robotique/AR (Boston Dynamics, Meta Reality Labs, etc.)

### Phase 3: SEO & Discoverability (Semaine 3-4)

#### A. Topics GitHub
Ajouter dans Settings → Topics:
```
computer-vision
depth-estimation
pytorch
rgbd
apple-silicon
cuda
xformers
robotics
3d-reconstruction
depth-refinement
```

#### B. Awesome lists PR
- [ ] PR à awesome-computer-vision
- [ ] PR à awesome-pytorch
- [ ] PR à awesome-robotics
- [ ] PR à awesome-3d-reconstruction

#### C. Papers With Code
- [ ] Lier le repo au paper ByteDance
- [ ] Ajouter benchmarks (si disponibles)

### Phase 4: Engagement communauté (Continu)

#### A. Issues templates
- [ ] Bug report template
- [ ] Feature request template
- [ ] Question template

#### B. Documentation interactive
- [ ] Ajout d'exemples Jupyter notebooks dans `examples/`
- [ ] Tutoriel vidéo YouTube (5-10 min)
- [ ] Blog post technique sur optimisations

#### C. Intégrations
- [ ] Hugging Face Spaces demo
- [ ] Gradio web interface
- [ ] Docker image (optionnel)

### Phase 5: Partenariats (Semaine 4+)

#### A. Contacter maintainers projets similaires
- DepthAnything
- MiDaS
- Marigold
- ZoeDepth

Proposer comparaisons, collaborations

#### B. Recherche académique
- Contacter auteurs du paper ByteDance
- Citer dans issues/discussions
- Proposer améliorations upstream

## 📈 Métriques de succès

**Semaine 1:**
- [ ] 50+ stars GitHub
- [ ] 100+ downloads PyPI
- [ ] 1 post Reddit >100 upvotes

**Semaine 2:**
- [ ] 150+ stars
- [ ] 500+ downloads
- [ ] Front page r/MachineLearning

**Semaine 4:**
- [ ] 300+ stars
- [ ] 2000+ downloads
- [ ] 5+ forks actifs

**3 mois:**
- [ ] 1000+ stars
- [ ] 10k+ downloads
- [ ] Featured dans awesome list
- [ ] Citation dans paper/blog

## 🎨 Assets à créer

### Priorité haute
1. **Demo GIF** (before/after depth refinement)
2. **Colab notebook** (essai en 1 clic)
3. **Comparison chart** (vs ByteDance original)
4. **Speed benchmark chart** (CPU/MPS/CUDA/xFormers)

### Priorité moyenne
5. Video tutorial (5 min)
6. Architecture diagram
7. Use case examples (robotics, AR, etc.)

### Priorité basse
8. Logo/branding
9. Website/landing page
10. Podcast interviews

## 💡 Messages clés (elevator pitch)

**30 secondes:**
"rgbd-depth transforms research code into production. One pip install gets you ByteDance's RGB-D depth refinement - 8% faster, Apple Silicon support, battle-tested."

**2 minutes:**
"ByteDance released amazing RGB-D depth research, but it required manual setup, had MPS bugs, and missed optimization opportunities. I packaged it properly: fixed the rendering bug, added xFormers support for 8% speedup, implemented mixed precision, and made it pip-installable. Now anyone can refine depth maps in production with one command."

## 🚫 Pièges à éviter

1. **Ne pas spam** - Max 1 post par communauté
2. **Être honnête** - Credit ByteDance clairement
3. **Pas de clickbait** - Métriques réelles uniquement
4. **Répondre vite** - Première 2h critique pour engagement
5. **Pas de self-promotion** sans valeur - Toujours donner avant de demander

## 📅 Timeline suggéré

**J1-3:** Créer assets visuels (GIFs, Colab)
**J4:** Post Reddit r/MachineLearning
**J5:** Post HN
**J6-7:** Twitter thread, LinkedIn
**J8-14:** Créer notebooks exemples, tutoriel
**J15:** Post Reddit r/computervision
**J16+:** PRs awesome lists, contacter maintainers

## 🔥 Quick wins immédiats

1. **Aujourd'hui:**
   - Ajouter badge PyPI version dans README
   - Ajouter badge PyPI downloads dans README
   - Créer 2-3 GIFs de démo

2. **Cette semaine:**
   - Colab notebook fonctionnel
   - Post Reddit r/MachineLearning
   - Twitter thread

3. **Ce mois:**
   - 3+ posts réseaux sociaux
   - Tutoriel vidéo
   - PRs awesome lists
