#!/bin/bash
# Builds marine_debris_kaggle.zip for upload to Kaggle.
# Run from the Dataset/ directory:
#   cd ~/Desktop/ML-Project\ /Dataset && bash create_kaggle_upload.sh

set -e
cd "$(dirname "$0")"

ZIP="$HOME/Desktop/marine_debris_kaggle.zip"
echo "Building $ZIP ..."
rm -f "$ZIP"

# Raw dataset (images + masks + annotations)
zip -r "$ZIP" \
  marine-debris-fls-datasets/md_fls_dataset/data/ \
  results/yolo_dataset/ \
  -x "*.DS_Store" -x "__pycache__/*" -x "*.pyc"

echo ""
echo "Done!  File: $ZIP"
du -sh "$ZIP"
echo ""
echo "Next steps:"
echo "  1. Go to kaggle.com → Datasets → New Dataset"
echo "  2. Upload $ZIP  (drag-and-drop)"
echo "  3. Name it 'marine-debris-fls'  →  Create"
echo "  4. Create a new Notebook → Add Data → select 'marine-debris-fls'"
echo "  5. Set accelerator: GPU T4 x2"
echo "  6. Upload kaggle_notebook.py as a script, or paste its contents"
echo "  7. In the notebook, verify DATA_ROOT and YOLO_ROOT paths, then Run All"
