# Autonomous Vision-Guided Robotic Book Retrieval and Management System
## Technical Internship Report Package

### Author & Academic Details
- **Student / Intern**: Joel V Babu (`joel-babu10`, `joelvbabuoneplus@gmail.com`)
- **Academic Supervisor**: Prof. Chiung-Hui Tsai
- **Institution**: Da-Yeh University
- **Department**: Department of Computer Science and Information Engineering
- **Internship Period**: August 13, 2026 – November 30, 2026

---

### Archive Contents
This package contains the complete publication-grade LaTeX source files, vector TikZ architecture diagrams, benchmark tables, and all 8 experimental figures:

1. `internship_report.tex`: Complete LaTeX document formatted according to standard university internship report guidelines (Chapters 1 to 11).
2. `figures/`: High-resolution figures and experimental live captures:
   - `training_curves_spine.png` (Figure 5.1: 100-epoch YOLO-OBB shelf spine loss and metric progression)
   - `confusion_matrix_spine.png` (Figure 5.2: Shelf spine OBB confusion matrix: 4,544 True Positives)
   - `training_curves_scattered.png` (Figure 5.3: 50-epoch tabletop scattered book loss and metric progression)
   - `confusion_matrix_scattered.png` (Figure 5.4: Tabletop scattered book confusion matrix: 884 True Positives)
   - `shelf_spine_live.jpg` (Figure 5.5: Live shelf spine detection on *JavaServer Pages* among *Linear Algebra* sibling editions by Leon, Nicholson, Bretscher)
   - `upright_mixed_live.jpg` (Figure 5.6: Live upright volume *Quality is Free* targeted beside tabletop literature in a mixed scene)
   - `scattered_book_live.jpg` (Figure 5.7: Live tabletop scattered retrieval of *Quality is Free* showing SAM contour and bound spine grasp edge)
   - `scattered_mixed_live.jpg` (Figure 5.8: Live mixed arrangement targeting of *JavaServer Pages* over *Linear Algebra Leon* with background shelf books)

---

### How to Compile
#### Option A: Overleaf (Recommended)
1. Log in to [Overleaf](https://www.overleaf.com).
2. Click **New Project** $\to$ **Upload Project**.
3. Select this ZIP file (`internship_report_package.zip`).
4. Set compiler to **pdfLaTeX** and click **Recompile**.

#### Option B: Local TeX Live / MiKTeX / MacTeX
Run `pdflatex` twice in your terminal to generate table of contents, figures, tables, and cross-references:
```bash
pdflatex internship_report.tex
pdflatex internship_report.tex
```
