from pprint import pprint
from sklearn.utils import all_estimators
from sklearn.tree import plot_tree
from functools import partial
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer

type_filter = [
    "classifier",
    "transformer",
]  # Which kind of estimators should be returned

arguments = {"type_filter": type_filter}
estimators = all_estimators(**arguments)
estimators = dict(estimators)

one_hot_encoder = estimators["OneHotEncoder"]

# Estas las quitamos porque hay armas que ciertos equipos no pueden comprar
variables_eliminar = [
    "t_weapon_famas",
    "ct_weapon_galilar",
    "t_weapon_usps",
    "ct_weapon_glock",
    "t_weapon_fiveseven",
    "t_weapon_p2000",
    "ct_weapon_tec9",
    "ct_weapon_mac10",
    "t_weapon_mp7",
    "t_weapon_mp9",
    "ct_weapon_g3sg1",
    "t_weapon_scar20",
    "t_weapon_mag7",
    "ct_weapon_sawedoff",
]

# Eliminamos las granadas
variables_eliminar = variables_eliminar + [
    "t_grenade_hegrenade",
    "t_grenade_smokegrenade",
    "t_grenade_flashbang",
    "t_grenade_molotovgrenade",
    "t_grenade_incendiarygrenade",
    "ct_grenade_hegrenade",
    "ct_grenade_smokegrenade",
    "ct_grenade_flashbang",
    "ct_grenade_molotovgrenade",
    "ct_grenade_incendiarygrenade",
    "ct_grenade_decoygrenade",
    "t_grenade_decoygrenade",
]


# Eliminamos las que se usan menos del 0.004
weapons_to_remove_ct = ['ct_weapon_bizon', 'ct_weapon_g3sg1', 'ct_weapon_m249', 'ct_weapon_mac10', 'ct_weapon_mp7', 'ct_weapon_negev', 'ct_weapon_nova', 'ct_weapon_r8revolver', 'ct_weapon_sawedoff', 'ct_weapon_scar20']
weapons_to_remove_t = ['t_weapon_m249', 't_weapon_negev', 't_weapon_scar20', 't_weapon_ssg08', 't_weapon_xm1014', 't_weapon_tec9']
variables_eliminar = variables_eliminar + weapons_to_remove_ct + weapons_to_remove_t

# Eliminamos por las correlaciones con otras variables
variables_eliminar = variables_eliminar + [
    "time_left",
    "t_armor",
    "ct_armor",
    "t_health",
    "ct_health",
]

# Dropper transformer
droper = make_column_transformer(
    ("drop", variables_eliminar),
    remainder="passthrough",
    verbose_feature_names_out=False,
)

def create_helmet(X):
    return X.assign(helmet=(X["t_helmets"] - X["ct_helmets"]))



# Crear un FunctionTransformer para aplicar la función a las columnas

creator = FunctionTransformer(create_helmet)

# Apply one-hot encode to 'map' column -> advanced pipeline
one_hot = make_column_transformer(
    (
        one_hot_encoder(sparse_output=False, handle_unknown="ignore"),
        ["bomb_planted", "map"],
    ),
    remainder="passthrough",
    verbose_feature_names_out=False,
)

# Para el árbol no tiene sentido estandarizar pero si puede tenerlo en un futuro si utilizamos KNN por lo que no lo utilizaremos en esta práctica
transformer4 = estimators["StandardScaler"]()