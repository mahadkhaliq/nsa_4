import rtamt

def create_stl_spec():
    """Create simple STL specification for NAS monitoring

    Monitors:
    - Accuracy should be above threshold
    - Energy should be below threshold
    """
    spec = rtamt.STLDiscreteTimeSpecification()
    spec.name = 'NAS Monitor'

    # Define variables
    spec.declare_var('accuracy', 'float')
    spec.declare_var('energy', 'float')

    # Simple STL: accuracy >= 0.6 and energy <= 100
    spec.spec = 'accuracy >= 0.6 and energy <= 100.0'

    try:
        spec.parse()
        return spec
    except rtamt.STLParseException as e:
        print(f'STL Parse Exception: {e}')
        return None

def evaluate_stl(accuracy, energy):
    """Evaluate STL specification for given accuracy and energy

    Returns:
        robustness: Positive means satisfied, negative means violated
    """
    spec = create_stl_spec()
    if spec is None:
        return None

    # Create trace
    trace = {
        'time': [0],
        'accuracy': [accuracy],
        'energy': [energy]
    }

    # Compute robustness
    robustness = spec.evaluate(['time', 'accuracy', 'energy'],
                               [trace['time'], trace['accuracy'], trace['energy']])

    return robustness[0][1]  # Return robustness value at time 0

def create_custom_stl(accuracy_threshold=0.6, energy_threshold=100.0):
    """Create custom STL specification with thresholds"""
    spec = rtamt.STLDiscreteTimeSpecification()
    spec.name = 'Custom NAS Monitor'

    spec.declare_var('accuracy', 'float')
    spec.declare_var('energy', 'float')

    spec.spec = f'accuracy >= {accuracy_threshold} and energy <= {energy_threshold}'

    try:
        spec.parse()
        return spec
    except rtamt.STLParseException as e:
        print(f'STL Parse Exception: {e}')
        return None
