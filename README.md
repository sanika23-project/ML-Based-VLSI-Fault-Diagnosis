# ML-Based VLSI Fault Diagnosis

Mini project: detect and diagnose stuck-at faults in a digital circuit
using a Random Forest classifier trained on simulated test-response
signatures.

## Circuit used
ISCAS-85 benchmark **c17**: 5 primary inputs, 2 primary outputs, 6 NAND
gates. It's the smallest standard benchmark circuit, which keeps the
simulation and dataset generation fast and easy to explain in a viva.

## How it works
1. `circuit_sim.py` simulates c17 logically and can inject a single
   stuck-at-0 or stuck-at-1 fault at any of the 6 internal gate outputs.
2. `generate_dataset.py` runs all 32 possible input test vectors through
   the circuit for each of the 13 classes (12 faults + fault-free),
   records the output response as a 64-bit "signature," and adds small
   random noise to create multiple training samples per class.
3. `train_model.py` trains a Random Forest classifier to map a 64-bit
   signature to the fault class that produced it, and reports accuracy,
   a classification report, and feature importance.
4. `streamlit_app.py` is a simple interface: pick a fault to inject (or
   none), click "Run Diagnosis," and see what the model predicts vs.
   the actual fault, with a confidence score.

## Running it

```bash
pip install -r requirements.txt

python generate_dataset.py   # creates dataset.csv
python train_model.py        # creates model.pkl, feature_importance.csv

streamlit run streamlit_app.py   # opens the demo in your browser
```

## Results (from the included run)
- Overall test accuracy: **~85%**
- Most fault classes are diagnosed perfectly (100% precision/recall)
- A few pairs of faults (e.g. N10_SA0 vs N19_SA0, N22_SA1 vs N23_SA1)
  are harder to tell apart because they produce nearly identical output
  signatures — this is a real, well-known phenomenon called **fault
  equivalence/masking** in VLSI testing, and is good material to
  discuss in your Results & Discussion slide and in the viva.

## Extending this for a stronger submission
- Swap in a bigger ISCAS-85 circuit (c432, c499) for more fault sites
- Inject faults at primary inputs too, not just gate outputs
- Try Decision Tree and Logistic Regression as a comparison against
  Random Forest (you already know these — easy "model comparison" slide)
- Replace the noise-based augmentation with faults injected under
  partial/reduced test sets, to show accuracy vs. test-set-size tradeoff

## Team split suggestion
- ECE teammate: understand and explain the circuit/fault model, verify
  circuit_sim.py logic matches real c17 netlist behavior
- CSE teammates: dataset generation, model training/tuning, Streamlit
  interface polish, and results visualization for the PPT
