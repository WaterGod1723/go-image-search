// Package detect provides ONNX-based single-object detection for the
// image-search pipeline.  It loads a trained DetectNet ONNX model and
// predicts one bounding box (cx, cy, w, h normalised) for an input image.
//
// The detector is used to crop the object region from a query image before
// perceptual-hash indexing, removing background-colour and text-interference
// noise (the TEST11 failure mode described in ANALYSIS.md).
package detect

import (
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"sync"

	"golang.org/x/image/draw"
	ort "github.com/yalue/onnxruntime_go"
)

const inputSize = 320

// Detector wraps an ONNX Runtime session for single-object detection.
type Detector struct {
	session *ort.DynamicAdvancedSession
	mu      sync.Mutex
}

// once guards the global onnxruntime environment so we only init it once.
var (
	once    sync.Once
	initErr error
)

// NewDetector loads an ONNX model from the given path. The onnxruntime
// shared library path must point to a valid libonnxruntime.dylib.
func NewDetector(modelPath, libPath string) (*Detector, error) {
	once.Do(func() {
		if libPath != "" {
			ort.SetSharedLibraryPath(libPath)
		}
		if err := ort.InitializeEnvironment(); err != nil {
			initErr = fmt.Errorf("onnxruntime init: %w", err)
		}
	})
	if initErr != nil {
		return nil, initErr
	}

	abs, err := filepath.Abs(modelPath)
	if err != nil {
		return nil, fmt.Errorf("resolve model path: %w", err)
	}
	s, err := ort.NewDynamicAdvancedSession(abs, []string{"image"}, []string{"bbox"}, nil)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}
	return &Detector{session: s}, nil
}

// Close destroys the underlying ONNX session.
func (d *Detector) Close() {
	if d.session != nil {
		d.session.Destroy()
	}
}

// Box is a normalised bounding box in [0, 1].
type Box struct {
	CX, CY, W, H float32
}

// PixelRect converts a normalised Box to pixel coordinates for the given
// image dimensions.
func (b Box) PixelRect(w, h int) image.Rectangle {
	x0 := int(math.Max(0, float64((b.CX-b.W/2)*float32(w))))
	y0 := int(math.Max(0, float64((b.CY-b.H/2)*float32(h))))
	x1 := int(math.Min(float64(w), float64((b.CX+b.W/2)*float32(w))))
	y1 := int(math.Min(float64(h), float64((b.CY+b.H/2)*float32(h))))
	if x1 <= x0 {
		x1 = x0 + 1
	}
	if y1 <= y0 {
		y1 = y0 + 1
	}
	return image.Rect(x0, y0, x1, y1)
}

// Detect runs the model on src and returns the predicted bounding box.
func (d *Detector) Detect(src image.Image) (Box, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Preprocess: resize to 320×320, NCHW float32 [0,1].
	data := preprocess(src)

	input, err := ort.NewTensor(ort.Shape{1, 3, inputSize, inputSize}, data)
	if err != nil {
		return Box{}, fmt.Errorf("create input tensor: %w", err)
	}
	defer input.Destroy()

	// Output: [1, 4] — let the session allocate it.
	outputs := []ort.Value{nil}
	if err := d.session.Run([]ort.Value{input}, outputs); err != nil {
		return Box{}, fmt.Errorf("run inference: %w", err)
	}
	out := outputs[0]
	defer out.Destroy()

	// Extract the 4 float32 values.
	outData, ok := extractFloat32(out)
	if !ok || len(outData) < 4 {
		return Box{}, fmt.Errorf("unexpected output shape")
	}
	return Box{CX: outData[0], CY: outData[1], W: outData[2], H: outData[3]}, nil
}

// DetectRect runs Detect and returns the result as a pixel-space image.Rectangle
// suitable for cropping the original image.
func (d *Detector) DetectRect(src image.Image) (image.Rectangle, error) {
	b := src.Bounds()
	box, err := d.Detect(src)
	if err != nil {
		return image.Rectangle{}, err
	}
	return box.PixelRect(b.Dx(), b.Dy()), nil
}

// Crop detects the object region in src and returns a sub-image of just that
// region.  The returned image is a sub-view of src; callers should not modify
// src while using the crop.
func (d *Detector) Crop(src image.Image) (image.Image, error) {
	rect, err := d.DetectRect(src)
	if err != nil {
		return nil, err
	}
	return CropImage(src, rect), nil
}

// preprocess resizes src to inputSize×inputSize and converts to NCHW float32
// normalised to [0, 1].
func preprocess(src image.Image) []float32 {
	b := src.Bounds()
	// Fast path: already the right size.
	if b.Dx() == inputSize && b.Dy() == inputSize {
		return imgToNCHW(src)
	}
	resized := image.NewRGBA(image.Rect(0, 0, inputSize, inputSize))
	draw.CatmullRom.Scale(resized, resized.Bounds(), src, b, draw.Over, nil)
	return imgToNCHW(resized)
}

// imgToNCHW converts an image to NCHW float32 [0,1] layout.
func imgToNCHW(img image.Image) []float32 {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	data := make([]float32, 3*w*h)
	plane := w * h
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, blue, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			idx := y*w + x
			data[idx] = float32(r) / 65535.0
			data[plane+idx] = float32(g) / 65535.0
			data[2*plane+idx] = float32(blue) / 65535.0
		}
	}
	return data
}

// CropImage returns the sub-image of src within rect, clamped to bounds.
func CropImage(src image.Image, rect image.Rectangle) image.Image {
	b := src.Bounds()
	rect = rect.Intersect(b)
	if rect.Empty() {
		return src
	}
	if sub, ok := src.(interface {
		SubImage(r image.Rectangle) image.Image
	}); ok {
		return sub.SubImage(rect)
	}
	// Fallback: copy pixels.
	dst := image.NewRGBA(rect)
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			dst.Set(x-rect.Min.X, y-rect.Min.Y, src.At(x, y))
		}
	}
	return dst
}

// DefaultModelPath returns the conventional path to the ONNX model relative
// to the project root.
func DefaultModelPath() string {
	// Try a few common locations.
	candidates := []string{
		"ml/models/detect.onnx",
		"models/detect.onnx",
		"detect.onnx",
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return "ml/models/detect.onnx"
}

// DefaultLibPath returns the conventional path to libonnxruntime.dylib.
func DefaultLibPath() string {
	candidates := []string{
		"ml/onnxruntime/lib/libonnxruntime.dylib",
		"onnxruntime/lib/libonnxruntime.dylib",
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	// Try DYLD_LIBRARY_PATH / pkg-config fallback
	return "libonnxruntime.dylib"
}

// extractFloat32 pulls float32 data from an ort.Value. We use a type switch
// on the concrete *ort.Tensor[float32] that our model produces.
func extractFloat32(v ort.Value) ([]float32, bool) {
	switch t := v.(type) {
	case *ort.Tensor[float32]:
		return t.GetData(), true
	}
	// Fallback: try underlying data via reflection-free cast
	// onnxruntime_go stores data in a Go slice; for float32 models the
	// concrete type is always *Tensor[float32].
	return nil, false
}
