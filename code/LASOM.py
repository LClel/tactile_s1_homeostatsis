import numpy as np
import scipy
from scipy.special import expit


def initial_activations(**args):
    """
    Using the dot product to calculate the response, calculates the relative contribution of afferent,
    lateral excitatory and inhibitory activations to the overall activation of each unit.

    args:
        affContribution (array) = 1D array of initial activation values calculated as a dot product of afferent weights
            and input
        inner_rad (int) = inner radius, used to identify all units within the excitatory radius of the unit
        outer_rad (int) = outer radius, used to identify all units within the inhibitory radius of the unit
        excitWeights (array) = 2D array containing lateral excitatory weights to for each unit to all other units
        inhibWeights (array) = 2D array containing lateral inhibitory weights to for each unit to all other units
        cortical_sheet_size (int) = number of cortical units
        output_X (array) = flattened x-axis of the cortical units
        output_Y (array) = flattened y-axis of the cortical units
        affFactor (float) = afferent weighting factor of the afferent contribution to unit activation
        excitFactor (float) = excitatory weighting factor of the excitatory contribution to unit activation
        inhibFactor (float) = inhibitory weighting factor of the inhibitory contribution to unit activation
    """

    affContribution = args.get('affContribution')
    inner_rad = args.get('inner_rad', 5)
    outer_rad = args.get('outer_rad', 15)
    excitWeights = args.get('excitWeights')
    inhibWeights = args.get('inhibWeights')

    cortical_sheet_size = args.pop('cortical_sheet_size', 900)
    output_X = args.get('output_X')
    output_Y = args.get('output_Y')

    affFactor = args.get('affFactor', .15)
    excitFactor = args.get('excitFactor', .17)
    inhibFactor = args.get('inhibFactor', .14)

    activation = np.zeros(cortical_sheet_size) # create empty vector to assign unit activation later
    inner = np.zeros((len(affContribution),len(affContribution)),dtype=bool)
    outer = np.zeros((len(affContribution),len(affContribution)),dtype=bool)

    for i in range(len(affContribution)):

        # select inner (excitatory) neighbourhood
        inner[i] = np.sqrt(((output_X - output_X[i]) ** 2 + ((output_Y - output_Y[i]) ** 2))) <= inner_rad

        # selects an outer (inhibitory) neighbourhood
        outer[i] = ((np.sqrt(((output_X - output_X[i]) ** 2 + ((output_Y - output_Y[i]) ** 2))) > inner_rad) & (
                    np.sqrt(((output_X - output_X[i]) ** 2 + ((output_Y - output_Y[i]) ** 2))) <= outer_rad))


    # calculate excitatory contribution to activation
    excitWeights[np.invert(inner)] = 0
    excitContribution = np.dot(affContribution,excitWeights.T)

    # calculate inhibitory contribution to activation
    inhibWeights[np.invert(outer)] = 0
    inhibContribution = np.dot(affContribution,inhibWeights.T)

    # calculate activation
    activation = (affContribution * affFactor) + (excitContribution * excitFactor)\
                              - (inhibContribution * inhibFactor)

    return activation, inner, outer



def homeostasis(**args):
    """
    Generates a threshold value for each cortical unit and compares the pre-calculated activation to it.
    Returns a vector of activations for each cortical unit.
    args:
        t (int) = input number
        beta (float) = smoothing parameter, determines the degree of smoothing in the calculation of the average activation
        lmbda (float) = homeostatic learning rate
        mu (float) = target average activation for cortical unit activity
        theta (dict) = dictionary of threshold values for each unit throughout all inputs
        av_act (dict) = dictionary of 'average' activation values for each unit throughout all inputs
        activation (array) = vector of unit activations calculated in initial_activations(**args)
    """

    t = args.get('t') # input number
    beta = args.get('beta', .991) # smoothing parameter
    lmbda = args.get('lmbda', .001) # homeostatic learning rate
    mu = args.get('mu', .024) # target average activation
    initTheta = args.get('initTheta', .10) # initial threshold activation
    theta = args.get('theta') # threshold values
    av_act = args.get('av_act')
    activation = args.get('activation')

    for i in range(len(activation)):

        if t == 0:
            activation[i] = activation[i] - initTheta

            av_act[i][t] = ((1 - beta) * activation[i]) + (
                        beta * mu)  # where mu here is the initialised value of av_act

            theta[i][t] = initTheta + (lmbda * (av_act[i][t] - mu))

            activation[i] = activation[i] - theta[i][t]

        else:
            activation[i] = activation[i] - theta[i][t - 1]

            av_act[i][t] = ((1 - beta) * activation[i]) + (beta * av_act[i][t - 1])

            theta[i][t] = theta[i][t - 1] + (lmbda * (av_act[i][t] - mu))

            activation[i] = activation[i] - theta[i][t]

    return activation, theta, av_act


def update_weights(**args):
    """
    Updates the afferent, lateral inhibitory and lateral excitatory weights. The change in weight is equal to
    the weight at the previous timestep plus the learning rate multiplied by the activation of unit i.
    args:
        activation (array) = vector of activation values following homeostatic threshold comparisons
        inner (dict) = dictionary containing booleans of units in the excitatory neighbourhood for each unit
        outer (dict) = dictionary containing booleans of units in the inhibitory neighbourhood for each unit
        affWeights (array) = array of afferent weights for each unit
        excitWeights (dict) = dictionary containing lateral excitatory weights to for each unit to all other units
        inhibWeights (dict) = dictionary containing lateral inhibitory weights to for each unit to all other units
        affAlpha (float) = learning rate for afferent weights
        excitAlpha (float) = learning rate for excitatory weights
        inhibAlpha (float)= learning rate for inhibitory weights
    """

    activation = args.get('activation')
    # inner = args.get('inner')
    # outer = args.get('outer')

    affWeights = args.get('affWeights').T
    inhibWeights = args.get('inhibWeights')
    excitWeights = args.get('excitWeights')

    affAlpha = args.get('affAlpha', .05)
    excitAlpha = args.get('excitAlpha', .01)
    inhibAlpha = args.get('inhibAlpha', .01)
    
    input_pre = args.get('presyn_input')

    x = np.size(affWeights,0)
    y = np.size(affWeights,1)
        
    # affWeight update
    affWeights_calc = affWeights + affAlpha * np.tile(activation,[y,1]).T * np.tile(input_pre,[x,1])
    affWeights = np.divide(affWeights_calc,np.tile(np.sum(affWeights_calc,0),[x,1]))
    
    # exciteWeights update
    excitWeights_calc = excitWeights + excitAlpha * np.tile(activation,[x,1]) * np.tile(activation,[x,1])
    excitWeights = np.divide(excitWeights_calc,np.tile(np.sum(excitWeights_calc,0),[x,1]))

    # inhibWeights update
    inhibWeights_calc = inhibWeights + inhibAlpha * np.tile(activation,[x,1]) * np.tile(activation,[x,1])
    inhibWeights = np.divide(inhibWeights_calc,np.tile(np.sum(inhibWeights_calc,0),[x,1]))
    
    return affWeights, excitWeights, inhibWeights

    ## Old code
    # for i in range(len(activation)):

    #     # multiple excitatory weights
    #     excitWeights[i][inner[i]] += excitAlpha * activation[i]

    #     inhibWeights[i][outer[i]] += inhibAlpha * activation[i]

    #     affWeights[i] += affAlpha * activation[i]
    
   # return affWeights.T, excitWeights, inhibWeights


# run lateral adaptive self organising map
def run_LASOM(inputs, **args):
    """
    Generates a weight array for a lateral connectivity self-organising map.
    Output array size is X*Y, where X is number of afferents and Y is
    number of cortical units. Each unit has a weight'strength of connection' to
    a cortical unit.
    args:
        seed (int): seed number (default: 1).
        sheet_size_a (int): cortical sheet size in 1st dimension (sqrt of number
            of cortical units for a square map). Default is 30
        sheet_size_b(int): cortical sheet size in 2nd dimension
            (sqrt of number of cortical units for a square map).
            Default is sheet_size_a
        inputs(array): np.array of predetermined responses to stimuli. Should be
            X*Y, where X is number of input dimensions (eg. afferents, joint angles) and
            Y is number of simulated responses of each of these dimensions to stimuli.
            Make array binary before inputting if needed.
        input_num (int): number of inputs presented to the SOM algorithm to create
            the map

        tag (str): name of the file to be saved. Should be the name of the folder
            in save location.
        path (str): path of save location (default: '', same folder currently in).
        affWeights (array): weights array, can include a pre-defined representation or previously
            learned map (eg. for plasticity retraining)

    Returns:
        affWeights: numpy array of weights
    """

    # input arguments
    seed = args.pop('seed', None)
    sheet_size_a = args.pop('sheet_size_a', 30)
    sheet_size_b = args.pop('sheet_size_b', sheet_size_a)
    input_num = args.pop('input_num', (sheet_size_a * sheet_size_b) * 10)
    affWeights = args.pop('w', None)

    # learning parameters
    affAlpha = args.pop('affAlpha', 0.1)  # afferent learning rate
    excitAlpha = args.pop('excitAlpha', 0.00)  # excitatory learning rate
    inhibAlpha = args.pop('inhibAlpha', 0.15)  # inhibitory

    # neighbourhood parameters
    inner = args.pop('inner', 5)  # excitatory neighbourhood
    outer = args.pop('outer', 10)  # inhibitory neighbourhood

    # homeostasis parameters
    theta = args.pop('initTheta', 0.01)  # first threshold value
    beta = args.pop('beta', .991)  # smoothing parameter

    # connection strength weighting factors
    affFactor = args.pop('affWeight', 1.5)  # afferent connection weighting factor
    inhibFactor = args.pop('inhibWeight', 1.4)  # inhibitory connection weighting factor
    excitFactor = args.pop('excitWeight', 1.7)  # excitatory connection weighting factor

    # create random seed
    if seed is not None:
        np.random.seed(seed)

    # In[Generate output cortical Sheet]

    # calculate total sheet size
    cortical_sheet_size = sheet_size_a * sheet_size_b

    # generate 2D sheet grid layout (cortical units)
    output_X, output_Y = np.meshgrid(np.linspace(1, sheet_size_a, sheet_size_a),
                                     np.linspace(1, sheet_size_b, sheet_size_b))
    output_X = output_X.flatten()
    output_Y = output_Y.flatten()

    hand_pop_length = inputs.shape[1]

    # In[Random weights]

    if affWeights is None:
        # initial random weights
        affWeights = np.random.rand(cortical_sheet_size, hand_pop_length)

        # normalise weights
        affWeights = np.divide(affWeights, sum(affWeights, 2))

    excitWeights = np.zeros((cortical_sheet_size,cortical_sheet_size))
    inhibWeights = np.zeros((cortical_sheet_size,cortical_sheet_size))

    for i in range(cortical_sheet_size):
        excitWeights[i] = np.random.rand(cortical_sheet_size)
        excitWeights[i] = excitWeights[i] / sum(excitWeights[i])
        inhibWeights[i] = np.random.rand(cortical_sheet_size)
        inhibWeights[i] = inhibWeights[i] / sum(inhibWeights[i])

    # In[Learning parameters]

    # In[Random inputs]

    # find number of input responses
    number_responses = inputs.shape[0]

    # vector of num_patterns length random integers, selected from number of responses
    random = np.random.choice(number_responses, input_num, replace=False)
    inputs = inputs[random, :]
    # In[Train network using Kohonen SOM]

    av_act = {}
    theta = {}
    for i in range(cortical_sheet_size):
        av_act[i] = np.zeros(input_num)
        theta[i] = np.zeros(input_num)

    # present one stimulus response per iteration to train the network
    for t in range(input_num):

        # calculate response from afferent connections
        affContribution = np.dot(affWeights, inputs[t, :].T)

        affWeights = affWeights.T

        activation, inner, outer = initial_activations(affContribution=affContribution, excitWeights=excitWeights,
                                                   inhibWeights=inhibWeights,cortical_sheet_size=cortical_sheet_size,
                                                   output_X=output_X,output_Y=output_Y)

        activation, theta, av_act = homeostasis(t=t, theta=theta, av_act=av_act, activation=activation)

        affWeights, excitWeights, inhibWeights = update_weights(activation=activation, inner=inner, outer=outer,
                                                                affWeights=affWeights, inhibWeights=inhibWeights,
                                                                excitWeights=excitWeights)

        affWeights = affWeights.T

    return affWeights


def singleunithomeostasis(**args):

    num = args.get('num')
    inputs = args.get('inputs')
    av_act = args.get('av_act')
    theta = args.get('theta')
    beta = args.get('beta',.991)
    mu = args.get('mu',.024)
    initTheta = args.get('initTheta',.01)
    act = args.get('act')
    lmbda = args.get('lmbda',.001)

    for t in range(num * 3):

        activation = inputs[t]

        if t == 0:
            av_act[t] = ((1 - beta) * activation) + (beta * mu)  # where mu here is the initialised value of av_act

            theta[t] = initTheta + (lmbda * (av_act[t] - mu))

            # raw activation = input - threshold
            act[t] = activation - theta[t]


        else:
            av_act[t] = ((1 - beta) * activation) + (beta * av_act[t - 1])

            theta[t] = theta[t - 1] + (lmbda * (av_act[t] - mu))

            # raw activation = input - threshold
            act[t] = activation - theta[t]


    return act,theta,av_act
