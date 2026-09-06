# You do not generate a new kernel per LLM

GLINT CUDA is **one** encode + **one** expand (+ later one H-TILE).  
Any BF16 model becomes a **pin** (hole+plug bytes). The kernel does not change.

```
model.safetensors (BF16)
        │  glint_pin.py  (libglint_encode.so)
        ▼
glint-pin/   schema glint_pin_v1
        │  ORCH_GLINT_PIN + GPU expand
        ▼
registry f32 (L0)  or resident hole+plug (L1)
        │  run_tdc_muon_course.sh  SEED=course.jsonl
        ▼
Muon LoRA steps on that course
```

Automatic driver: `bash scripts/glint_auto.sh --src BF16 --course tdc_v2 --steps 8 --orch /path/to/training_orchestrator`

Courses: `tdc_v2` `tdc_v1` `a` `b` or a `.jsonl` path.

