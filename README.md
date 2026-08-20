# NewGen6 Pojav/Fabric Mod

Target: Minecraft 1.21.11, Fabric, client-side.

## Current state
- C toggles NewGen6 AI ON/OFF.
- Shows an action-bar ON/OFF indicator.
- Does NOT yet send model actions to Minecraft.
- The ONNX runtime JAR must be supplied separately as `libs/onnxruntime.jar`.

## Model files
Put these in `src/main/resources/newgen6/`:
- newgen6_full.onnx
- combat_move_model.json

## Important
The model expects `[batch, 96, 156]` and exposes 29 action heads. The exact feature construction/normalization must be implemented before inference is enabled.
