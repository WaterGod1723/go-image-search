package attnnet

// Trainer wraps a model plus its Adam optimizer so commands can run single
// samples, accumulate a mini-batch and apply one optimizer step.
type Trainer struct {
	m   *Model
	opt *adamState
}

// NewTrainer builds a Trainer around m.
func NewTrainer(m *Model) *Trainer {
	return &Trainer{m: m, opt: newAdam(m)}
}

// Model returns the underlying model (e.g. to Save it).
func (t *Trainer) Model() *Model { return t.m }

// Step runs forward + backward for one (x, mask) sample and accumulates the
// gradients into the optimizer. It returns the CE loss and the gate BCE loss.
func (t *Trainer) Step(x []uint8, mask []uint8) (ce, gate float64) {
	a, ce, gl := t.m.forward(x, mask)
	_, gl2 := t.m.backward(a, x, mask, t.opt.gs)
	_ = gl
	return ce, gl2
}

// Apply performs one Adam update over the gradients accumulated since the last
// Apply/Reset, using invN = 1/batchSize as the gradient scale.
func (t *Trainer) Apply(invN, lr float64) { t.opt.apply(invN, lr) }

// Reset zeroes the accumulated gradients.
func (t *Trainer) Reset() { t.opt.reset() }

// Eval runs the model without gradients and returns the softmax attention
// distribution and losses for one sample.
func (t *Trainer) Eval(x, mask []uint8) (attn []float64, ce, gate float64) {
	a, ce, gl := t.m.forward(x, mask)
	return softmaxVec(a.logits), ce, gl
}
