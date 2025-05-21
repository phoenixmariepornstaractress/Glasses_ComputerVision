# Glasses Try-On App

A real-time virtual try-on application that overlays different pairs of glasses onto a user's face using OpenCV and computer vision techniques. The app detects faces using Haar Cascades and provides interactive features for trying, scoring, and selecting the best-fitting glasses based on facial proportions.

## Features

* Real-time webcam-based face detection
* Overlay glasses images accurately on detected faces
* Sidebar with selectable glasses thumbnails
* Automatic scoring system to rank glasses based on face shape
* Snapshot saving functionality
* Face data logging and auto-selection of the best matching glasses
* Multiple face detection with centering tips
* Custom sidebar interaction with mouse clicks

## Getting Started

### Prerequisites

* Python 3.7+
* OpenCV
* Pillow
* NumPy

Install dependencies using:

```bash
pip install opencv-python numpy Pillow
```

### Usage

Place your `glasses*.png` images (with transparent backgrounds) in the project directory. Then run:

```bash
python main.py
```

### Controls

* `q`: Quit the application
* `n`: Next pair of glasses
* `s`: Save a snapshot
* `o`: Toggle glasses overlay
* `b`: Automatically select the best matching glasses

## Glasses Image Requirements

* Format: PNG with transparency (RGBA)
* Filenames should start with "glasses" (e.g., `glasses1.png`, `glasses2.png`)
* Images should be cropped to fit standard aspect ratios for virtual glasses

## Contributing

Contributions are welcome! If you'd like to add new features, improve detection algorithms, or enhance UI, please:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a Pull Request

### Suggested Improvements for Contributors

* Integrate DNN-based face landmark detection for more accurate glasses placement
* Add GUI controls with Tkinter or PyQt
* Implement machine learning-based glasses recommendations
* Add support for different face shapes
* Support image upload mode (for non-live preview)

## License

This project is licensed under the MIT License.

---

Happy coding, and thank you for contributing to Glasses Try-On App!
