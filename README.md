I am working on a production scheduling and indent allocation system for a dairy plant, initially focusing on curd (Cup, Bucket) and Mishti Doi. This is a real-world operations research problem and I want to implement a Python-based MVP to simulate and optimize production planning before porting the logic to Excel/VBA for deployment.

### 🧭 Objective:
Maximize total production output while respecting constraints on tank capacity, shift durations, CIP/setup times, SKU priorities, and feasibility rules.

### 📥 Input Tables (all will be read from CSV or Excel):
1. USER_INDENT: SKU-ID, IND-QTY-MIN, IND-QTY-MAX, PRIORITY (lower number = higher priority)
2. SKU_CONFIG: SKU-ID, PRODUCT CATEGORY, VARIANT, LINE
3. LINE_SKU: SKU-ID, LINE, PEAK FLOW, TARGET EFFICIENCY, SETUP TIME
4. LINE_CONFIG: LINE, CIP CIRCUIT, CIP TIME, ACTIVE
5. TANK_CONFIG: TANK-ID, CAPACITY, CIP CIRCUIT, CIP TIME, ACTIVE
6. TANK_PROD: TANK-ID, PRODUCT CATEGORY, VARIANT
7. SHIFT_CONFIG: SHIFT-ID, START-TIME, END-TIME, DURATION, BREAK, PRODUCTIVE DURATION
8. SPL_CONSTRAINTS: SKU-A, SKU-B, RELN (DEPENDENT, INDEPENDENT, EXCLUSIVE)
9. CIP_CONFIG (OPTIONAL): CIP-CIRCUIT, CONNECTION (PARALLEL/SERIES)

### ⚙️ Core Constraints:
- Every SKU must be produced on compatible lines and tanks.
- Every line changeover adds SETUP TIME.
- Variant change in tanks requires a CIP cycle.
- After a tank is emptied, it must undergo CIP before reuse.
- Intra-shift tank CIP is allowed but discouraged.
- Setup/CIP times consume productive shift durations.
- Each SKU has a Min/Max indent request. If not fulfilled, system should attempt to fill as much as possible.
- Unfilled space in shifts should be used to maximize production of other feasible SKUs.

### ⚡ Goal:
Build a Python MVP that:
1. Reads all these tables
2. Sorts USER_INDENT by PRIORITY
3. Allocates SKU production to shift-line-tank combinations using heuristics (OR-inspired)
4. Maximizes total production
5. Writes out a final schedule with SKU, Line, Tank, Shift, Qty, Flags for CIP/Setup

Please help me step-by-step starting from loading and preprocessing these tables, all the way to scheduling logic.

