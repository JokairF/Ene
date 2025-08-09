---

```markdown
# Ene – Assistant Virtuel Local

[![Repo GitHub](https://img.shields.io/badge/GitHub-Ene-181717?logo=github)](https://github.com/JokairF/Ene)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)

Ene est un **assistant virtuel francophone** fonctionnant entièrement **en local** grâce à `llama-cpp-python`.  
Son caractère et son style de réponse sont définis par la **Bible du Projet Ene**, offrant une expérience conversationnelle **immersive et personnalisée**.

---

## 🚀 Fonctionnalités

- **Mode Chat avec personnalité** : conversation fluide, ton naturel et engageant.
- **Résumé et analyse** : capacité à synthétiser des informations complexes.
- **Streaming SSE** : réponses envoyées en temps réel.
- **Historique de session** pour conserver le contexte.
- **Mode CPU ou GPU** selon votre configuration.
- **API REST** pour intégration dans d'autres applications.

---

## 📂 Structure du projet

```

Ene/
├── app/
│   ├── main.py         # Point d'entrée FastAPI
│   ├── llm.py          # Gestion du modèle LLaMA
│   ├── persona.py      # Personnalité et instructions système
│   └── ...
├── requirements.txt
├── Bible\_du\_Projet\_Ene.pdf
├── README.md
└── ...

````

---

## 📦 Installation

### 1️⃣ Cloner le dépôt

```bash
git clone https://github.com/JokairF/Ene.git
cd Ene
````

### 2️⃣ Créer un environnement virtuel

```bash
python -m venv .venv
# Activer l'environnement
# Windows :
.venv\Scripts\activate
# Linux/macOS :
source .venv/bin/activate
```

### 3️⃣ Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚡ Exécution

### Lancer en mode développement

```bash
uvicorn app.main:app --reload
```

📍 Accès API : [http://localhost:8000](http://localhost:8000)
📍 Documentation interactive Swagger : [http://localhost:8000/docs](http://localhost:8000/docs)
📍 Documentation ReDoc : [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🖥 Mode CPU et GPU

* **CPU (par défaut)** → fonctionne partout.
* **GPU (CUDA)** → nécessite :

  * Drivers NVIDIA récents
  * CUDA Toolkit installé (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`)
  * Compilation de `llama-cpp-python` avec support CUDA :

    ```bash
    pip uninstall llama-cpp-python
    pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir \
      --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=86 \
      --config-settings=cmake.define.LLAMA_CUBLAS=ON \
      --config-settings=cmake.define.CMAKE_BUILD_TYPE=Release
    ```

Pour forcer le CPU :

```bash
set LLAMA_CPP_GPU=0  # Windows
export LLAMA_CPP_GPU=0  # Linux/macOS
```

---

## 📡 API – Endpoints principaux

### **1. Chat en streaming**

`POST /chat/stream`

**Requête :**

```json
{
  "session_id": "demo-1",
  "message": "Salut Ene, peux-tu me résumer le projet ?"
}
```

**Réponse :**
Flux **SSE** avec le texte envoyé au fur et à mesure.

---

## 🎭 Personnalité d’Ene

D’après la **Bible du Projet Ene**, l’assistant :

* S’exprime en français, ton **chaleureux, naturel et engageant**.
* Peut être à la fois **informatif, complice et proactif**.
* Répond de manière **complète et contextuelle**, sans se limiter à une phrase courte.
* Conserve une **identité constante** et reconnaissable tout au long de la conversation.

Exemple :

> *"Oh, je vois où tu veux en venir 😏 !
> Laisse-moi t’expliquer ça simplement, mais sans rien oublier d’important..."*

---

## 🛠 Contribution

1. Forker le dépôt
2. Créer une branche :

   ```bash
   git checkout -b feature/ma-feature
   ```
3. Commit et push :

   ```bash
   git commit -m "Ajout d'une nouvelle fonctionnalité"
   git push origin feature/ma-feature
   ```
4. Créer une Pull Request

---

## 📜 Licence

Projet interne – **tous droits réservés**.
Contact : [GitHub Issues](https://github.com/JokairF/Ene/issues)

---