import rtamt

def create_stl_spec(quality_constraint=0.70, energy_constraint=50.0):
    spec = rtamt.StlDiscreteTimeSpecification()
    spec.name = 'ApproxAI NAS Monitor'

    spec.declare_var('accuracy', 'float')
    spec.declare_var('energy', 'float')

    spec.spec = f'accuracy >= {quality_constraint} and energy <= {energy_constraint}'

    try:
        spec.parse()
        return spec
    except rtamt.RTAMTException as e:
        print(f'STL Parse Exception: {e}')
        return None

def evaluate_stl(accuracy, energy, quality_constraint=0.70, energy_constraint=50.0):
    spec = create_stl_spec(quality_constraint, energy_constraint)
    if spec is None:
        return None

    spec.pastify()
    robustness = spec.update(0, [('accuracy', accuracy), ('energy', energy)])

    return robustness

def check_pareto_optimal(results):
    pareto_indices = []

    for i, result_i in enumerate(results):
        if result_i['approx_accuracy'] is None or result_i['energy'] is None:
            continue

        is_dominated = False
        for j, result_j in enumerate(results):
            if i == j or result_j['approx_accuracy'] is None or result_j['energy'] is None:
                continue

            if (result_j['approx_accuracy'] > result_i['approx_accuracy'] and
                result_j['energy'] < result_i['energy']):
                is_dominated = True
                break

        if not is_dominated:
            pareto_indices.append(i)

    return pareto_indices

def create_custom_stl(accuracy_threshold=0.6, energy_threshold=100.0):
    return create_stl_spec(accuracy_threshold, energy_threshold)
