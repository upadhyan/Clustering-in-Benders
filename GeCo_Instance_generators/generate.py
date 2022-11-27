from .packing import tang_instance_packing

from .independent_set import barabasi_albert_instance as indep_set_BA, independent_set
from .independent_set import gasse_instance as indep_set_Gasse

from .knapsack import yang_instance as knapsack_yang_instance
from .knapsack import uncorrelated as knapsack_uncorrelated
from .knapsack import uncorrelated_with_similar_weights as knapsack_uncorrelated_with_similar_weights

from .set_covering import yang_instance as set_cover_Yang
from .set_covering import sun_instance as set_cover_Sun
from .set_covering import gasse_instance as set_cover_Gasse

from .set_packing import yang_instance as set_packing_Yang

from .production_planning import tang_instance as production_planning_Tang

from .scheduling import heinz_instance as scheduling_heinz
from .scheduling import hooker_instance as scheduling_hooker

from .comb_auction import gasse_instance as combinatorial_auction_gasse
from .facility_location import cornuejols_instance as facility_location_cornuejols

from .max_cut import tang_instance as max_cut_tang


def gen_MIP(instance_type,instance_config,seed):


    if instance_type == "max_cut":
        ip = max_cut_tang(n= instance_config['n'], m= instance_config['m'], seed =  seed)

    if instance_type == "comb_auction":
        ip = combinatorial_auction_gasse(n_items=instance_config['n_items'],n_bids=instance_config['n_bids'],seed = seed)

    if instance_type == "facility_location":
        ip = facility_location_cornuejols(n_customers=instance_config['n_customers'], n_facilities=instance_config['n_facilities'], ratio=instance_config['ratio'],seed = seed)


    if instance_type == "independent_set":
        if instance_config['type'] == "Gasse":
            ip = indep_set_Gasse(n = instance_config['n'], p = instance_config['p'], seed = seed)
        elif instance_config['type'] == "barabasi_albert":
            ip = indep_set_BA(n = instance_config['n'],m= instance_config['m'], seed = seed)

    if instance_type == "knapsack":
        if instance_config['type'] == "yang":
            ip = knapsack_yang_instance(n = instance_config['n'])
        if instance_config['type'] == "uncorrelated":
            ip = knapsack_uncorrelated(n = instance_config['n'], c = instance_config['c'], R=1000, seed=seed)
        if instance_config['type'] == "uncorrelated_with_similar_weights":
            ip = knapsack_uncorrelated_with_similar_weights(n = instance_config['n'], c = instance_config['c'], R=1000, seed=seed)

    if instance_type == "packing":
        if instance_config['type'] == "integer":
            #instance_type = "packing-I"
            ip = tang_instance_packing(n= instance_config['n'], m= instance_config['m'], binary = False, seed =  seed)
        if instance_config['type'] == "binary":
            #instance_type = "packing-B"
            ip = tang_instance_packing(n= instance_config['n'], m= instance_config['m'], binary = True, seed = seed)

    if instance_type == "production_planning":
        ip = production_planning_Tang(T = instance_config['T'],seed = seed)
    
    if instance_type == "scheduling":
        
        if instance_config['type'] == "hooker":
            ip = scheduling_heinz(number_of_facilities = instance_config['number_of_facilities'],number_of_tasks = instance_config['number_of_tasks'],seed = seed)

        if instance_config['type'] == "heinz":
            ip = scheduling_heinz(number_of_facilities = instance_config['number_of_facilities'],number_of_tasks = instance_config['number_of_tasks'],seed = seed)

    if instance_type == "set_covering":

        if instance_config['type'] == "Gasse":
            ip = set_cover_Gasse(nrows =instance_config['m'],ncols= instance_config['n'],density = instance_config['density'], max_coef = instance_config['max_coeff'])
        if instance_config['type'] == "sun":
            ip = set_cover_Sun(n=instance_config['n'],m=instance_config['m'],seed = seed)
        if instance_config['type'] == "yang":
            ip = set_cover_Yang(m=instance_config['m'],seed=seed)

    if instance_type == "set_packing":
        ip = set_packing_Yang(m=instance_config['m'],seed=seed)

    return ip