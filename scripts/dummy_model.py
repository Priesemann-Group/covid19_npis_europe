import covid19_npis


""" # Data Retrieval
	Retries some dummy/test data
"""
# Fixed R matrix for now one country one age group
I_new = covid19_npis.test_data.simple_new_I()
age_groups = 4
""" # Construct pymc4 model
"""


@pm.model
def test_model(data):

    # Create I_0
    shape_I_0 = len(age_groups)
    I_0 = yield pm.HalfCauchy(loc=[10] * shape_I_0, name="I_0")

    # Create Reproduktion number for every age group
    shape_R = len(age_groups)
    R = yield pm.Normal(loc=[2] * shape_R, scale=2.5, name="R_age_groups")

    # Create Contact matrix
    shape_C = 1
    C = yield pm.LKJ(
        dimension=len(age_groups),
        concentration=[2] * shape_C,  # eta
        name="Contact_matrix",
    )

    new_cases = covid19_npis.model.InfectionModel(
        N=10e5, I_0=I_0, R_t=R, g=None, l=16  # default value
    )
