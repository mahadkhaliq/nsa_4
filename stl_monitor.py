import rtamt

def create_stl_spec(quality_constraint=0.70, energy_constraint=50.0):
    """Create STL specification based on approxAI paper constraints

    Based on XAI-Gen methodology from approxAI research:
    - Quality Constraint (Qc): Minimum acceptable accuracy
    - Energy Constraint (Ec): Maximum acceptable energy (mJ)

    The paper uses these constraints to explore Pareto-optimal
    approximate DNN designs.

    Args:
        quality_constraint: Minimum accuracy threshold (Qc), default 0.70
        energy_constraint: Maximum energy in mJ (Ec), default 50.0

    Returns:
        spec: STL specification or None if parsing fails
    """
    spec = rtamt.StlDiscreteTimeSpecification()
    spec.name = 'ApproxAI NAS Monitor'

    # Define variables
    spec.declare_var('accuracy', 'float')
    spec.declare_var('energy', 'float')

    # STL: (accuracy >= Qc) AND (energy <= Ec)
    # Following approxAI paper's constraint model
    spec.spec = f'accuracy >= {quality_constraint} and energy <= {energy_constraint}'

    try:
        spec.parse()
        return spec
    except rtamt.RTAMTException as e:
        print(f'STL Parse Exception: {e}')
        return None

def evaluate_stl(accuracy, energy, quality_constraint=0.70, energy_constraint=50.0):
    """Evaluate STL specification with approxAI paper constraints

    Based on approxAI paper methodology:
    - Qc (quality constraint): minimum accuracy threshold
    - Ec (energy constraint): maximum energy threshold

    The robustness value indicates how well constraints are satisfied:
    - Positive: Constraints satisfied (larger = more margin)
    - Negative: Constraints violated (more negative = worse violation)

    Args:
        accuracy: Model accuracy (0.0 to 1.0)
        energy: Energy consumption in mJ
        quality_constraint: Qc threshold (default: 0.70)
        energy_constraint: Ec threshold in mJ (default: 50.0)

    Returns:
        robustness: Positive if satisfied, negative if violated
    """
    spec = create_stl_spec(quality_constraint, energy_constraint)
    if spec is None:
        return None

    # Update with single timestep
    spec.pastify()
    robustness = spec.update(0, [('accuracy', accuracy), ('energy', energy)])

    return robustness

def check_pareto_optimal(results):
    """Identify Pareto-optimal architectures (approxAI methodology)

    An architecture is Pareto-optimal if no other architecture
    has both higher accuracy AND lower energy (dominates it).

    This implements the multi-objective optimization approach
    from the approxAI paper.

    Args:
        results: List of result dicts with 'approx_accuracy' and 'energy'

    Returns:
        pareto_indices: List of indices of Pareto-optimal results
    """
    pareto_indices = []

    for i, result_i in enumerate(results):
        if result_i['approx_accuracy'] is None or result_i['energy'] is None:
            continue

        is_dominated = False
        for j, result_j in enumerate(results):
            if i == j or result_j['approx_accuracy'] is None or result_j['energy'] is None:
                continue

            # Check if j dominates i (higher accuracy AND lower energy)
            if (result_j['approx_accuracy'] > result_i['approx_accuracy'] and
                result_j['energy'] < result_i['energy']):
                is_dominated = True
                break

        if not is_dominated:
            pareto_indices.append(i)

    return pareto_indices

def create_custom_stl(accuracy_threshold=0.6, energy_threshold=100.0):
    """Create custom STL specification with thresholds

    Legacy function - prefer create_stl_spec() with approxAI constraints
    """
    return create_stl_spec(accuracy_threshold, energy_threshold)
