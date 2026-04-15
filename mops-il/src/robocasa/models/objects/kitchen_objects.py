import os

import robocasa

BASE_ASSET_ZOO_PATH = os.path.join(robocasa.models.assets_root, "objects")


# Constant that contains information about each object category. These will be used to generate the ObjCat classes for each category
OBJ_CATEGORIES = {
    "liquor": {
        "types": ("drink", "alcohol"),
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "model_folders": ["aigen_objs/alcohol"],
            "scale": 1.50,
        },
        "objaverse": {
            "model_folders": ["objaverse/alcohol"],
            "scale": 1.35,
        },
    },
    "apple": {
        "types": ("fruit"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": True,
        "freezable": False,
        "aigen": {
            "scale": 1.0,
        },
        "objaverse": {
            "scale": 0.90,
        },
    },
    "avocado": {
        "types": ("vegetable"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 0.90,
        },
        "objaverse": {
            "scale": 0.90,
        },
    },
    "bagel": {
        "types": ("bread_food"),
        "graspable": False,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.2,
        },
        "objaverse": {
            "exclude": [
                "bagel_8",
            ],
        },
    },
    "bagged_food": {
        "types": ("packaged_food"),
        "graspable": False,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": 1.1,
        },
        "objaverse": {
            "exclude": [
                "bagged_food_12",
            ],
        },
    },
    "baguette": {
        "types": ("bread_food"),
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": 1.35,
        },
        "objaverse": {
            "exclude": [
                "baguette_3",  # small holes on ends
            ],
        },
    },
    "banana": {
        "types": ("fruit"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.10,
        },
        "objaverse": {
            "scale": 0.95,
        },
    },
    "bar": {
        "types": ("packaged_food"),
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": [1.25, 1.25, 1.75],
        },
        "objaverse": {
            "scale": [0.75, 0.75, 1.2],
            "exclude": [
                "bar_1",  # small holes scattered
            ],
        },
    },
    "bar_soap": {
        "types": ("cleaner"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": [1.25, 1.25, 1.40],
        },
        "objaverse": {
            "scale": [0.95, 0.95, 1.05],
            "exclude": ["bar_soap_2"],
        },
    },
    "beer": {
        "types": ("drink", "alcohol"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.30,
        },
        "objaverse": {"scale": 1.15},
    },
    "bell_pepper": {
        "types": ("vegetable"),
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "aigen": {
            "scale": 1.0,
        },
        "objaverse": {
            "scale": 0.75,
        },
    },
    "bottled_drink": {
        "types": ("drink"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": 1.25,
        },
        "objaverse": {},
    },
    "bottled_water": {
        "types": ("drink"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": 1.30,
        },
        "objaverse": {
            "scale": 1.10,
            "exclude": [
                "bottled_water_0",  # minor hole at top
                "bottled_water_5",  # causing error. eigenvalues of mesh inertia violate A + B >= C
            ],
        },
    },
    "bowl": {
        "types": ("receptacle", "stackable"),
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.75,
        },
        "objaverse": {
            "scale": 2.0,
            "exclude": [
                "bowl_21",  # can see through from bottom of bowl
            ],
        },
    },
    "boxed_drink": {
        "types": ("drink"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": 1.1,
        },
        "objaverse": {
            "scale": 0.80,
            "exclude": [
                "boxed_drink_9",  # hole on bottom
                "boxed_drink_6",  # hole on bottom
                "boxed_drink_8",  # hole on bottom
            ],
        },
    },
    "boxed_food": {
        "types": ("packaged_food"),
        "graspable": True,
        "washable": False,
        "microwavable": True,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": 1.25,
        },
        "objaverse": {
            "scale": 1.1,
            "exclude": [
                "boxed_food_5",  # causing error. eigenvalues of mesh inertia violate A + B >= C
            ],
            # exclude=[
            #     "boxed_food_5",
            #     "boxed_food_3", "boxed_food_1", "boxed_food_6", "boxed_food_11", "boxed_food_10", "boxed_food_8", "boxed_food_9", "boxed_food_7", "boxed_food_2", # self turning due to single collision geom
            # ],
        },
    },
    "bread": {
        "types": ("bread_food"),
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": [0.80, 0.80, 1.0],
        },
        "objaverse": {
            "scale": [0.70, 0.70, 1.0],
            "exclude": ["bread_22"],
        },  # hole on bottom
    },
    "broccoli": {
        "types": ("vegetable"),
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "aigen": {
            "scale": 1.35,
        },
        "objaverse": {
            "scale": 1.25,
            "exclude": [
                "broccoli_2",  # holes on one part
            ],
        },
    },
    "cake": {
        "types": ("sweets"),
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": 0.8,
        },
        "objaverse": {
            "scale": 0.8,
        },
    },
    "can": {
        "types": ("drink"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "aigen": {},
        "objaverse": {
            "exclude": [
                "can_10",  # hole on bottom
                "can_5",  # causing error: faces of mesh have inconsistent orientation.
            ],
        },
    },
    "candle": {
        "types": ("decoration"),
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.5,
        },
        "objaverse": {
            "exclude": [
                "candle_11",  # hole at bottom
                # "candle_2", # can't see from bottom view angle
                # "candle_15", # can't see from bottom view angle
            ],
        },
    },
    "canned_food": {
        "types": ("packaged_food"),
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": 1.15,
        },
        "objaverse": {
            "scale": 0.90,
            "exclude": [
                "canned_food_7",  # holes at top and bottom
            ],
        },
    },
    "carrot": {
        "types": ("vegetable"),
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "aigen": {
            "scale": 1.25,
        },
        "objaverse": {},
    },
    "cereal": {
        "types": ("packaged_food"),
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.15,
        },
        "objaverse": {
            # exclude=[
            #     "cereal_2", "cereal_5", "cereal_13", "cereal_3", "cereal_9", "cereal_0", "cereal_7", "cereal_4", "cereal_8", "cereal_12", "cereal_11", "cereal_1", "cereal_6", "cereal_10", # self turning due to single collision geom
            # ]
        },
    },
    "cheese": {
        "types": ("dairy"),
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "aigen": {
            "scale": 1.0,
        },
        "objaverse": {
            "scale": 0.85,
        },
    },
    "chips": {
        "types": ("packaged_food"),
        "graspable": False,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.5,
        },
        "objaverse": {
            "exclude": [
                "chips_12",  # minor hole at bottom of bag
                # "chips_2", # a weird texture at top/bottom but keeping this
            ]
        },
    },
    "chocolate": {
        "types": ("sweets"),
        "graspable": False,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": [1.0, 1.0, 1.35],
        },
        "objaverse": {
            "scale": [0.80, 0.80, 1.20],
            "exclude": [
                # "chocolate_2", # self turning due to single collision geom
            ],
        },
    },
    "coffee_cup": {
        "types": ("drink"),
        "graspable": True,
        "washable": False,
        "microwavable": True,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.35,
        },
        "objaverse": {
            "exclude": [
                "coffee_cup_18",  # can see thru top
                "coffee_cup_5",  # can see thru from bottom side
                "coffee_cup_19",  # can see thru from bottom side
            ],
        },
    },
    "condiment_bottle": {
        "types": ("condiment"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.35,
            "model_folders": ["aigen_objs/condiment"],
        },
        "objaverse": {
            "scale": 1.05,
            "model_folders": ["objaverse/condiment"],
        },
    },
    "corn": {
        "types": ("vegetable"),
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "aigen": {
            "scale": 1.5,
        },
        "objaverse": {"scale": 1.05},
    },
    "croissant": {
        "types": ("pastry"),
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": 0.90,
        },
        "objaverse": {
            "scale": 0.90,
        },
    },
    "cucumber": {
        "types": ("vegetable"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": 1.1,
        },
        "objaverse": {},
    },
    "cup": {
        "types": ("receptacle", "stackable"),
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.35,
        },
        "objaverse": {},
    },
    "cupcake": {
        "types": ("sweets"),
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": 0.90,
        },
        "objaverse": {
            "exclude": [
                "cupcake_0",  # can see thru bottom
                "cupcake_10",  # can see thru bottom,
                "cupcake_1",  # very small hole at bottom
            ]
        },
    },
    "cutting_board": {
        "types": ("receptacle"),
        "graspable": False,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 2.0,
        },
        "objaverse": {
            "scale": 1.35,
            "exclude": [
                "cutting_board_14",
                "cutting_board_3",
                "cutting_board_10",
                "cutting_board_6",  # these models still modeled with meshes which should work most of the time, but excluding them for safety
            ],
        },
    },
    "donut": {
        "types": ("sweets", "pastry"),
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": 1.5,
        },
        "objaverse": {
            "scale": 1.15,
        },
    },
    "egg": {
        "types": ("dairy"),
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "aigen": {
            "scale": 1.15,
        },
        "objaverse": {
            "scale": 0.85,
        },
    },
    "eggplant": {
        "types": ("vegetable"),
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "aigen": {
            "scale": 1.30,
        },
        "objaverse": {"scale": 0.95},
    },
    "fish": {
        "types": ("meat"),
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "aigen": {
            "scale": [1.35, 1.35, 2.0],
        },
        "objaverse": {
            "scale": [1.0, 1.0, 1.5],
        },
    },
    "fork": {
        "types": ("utensil"),
        "graspable": False,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": False,
        "aigen": {
            "scale": 1.75,
        },
        "objaverse": {},
    },
    "garlic": {
        "types": ("vegetable"),
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "aigen": {
            "scale": 1.3,
        },
        "objaverse": {"scale": 1.10, "exclude": ["garlic_3"]},  # has hole on side
    },
    "hot_dog": {
        "types": ("cooked_food"),
        "graspable": True,
        "washable": False,
        "microwavable": True,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": 1.4,
        },
        "objaverse": {},
    },
    "jam": {
        "types": ("packaged_food"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": 1.05,
        },
        "objaverse": {
            "scale": 0.90,
        },
    },
    "jug": {
        "types": ("receptacle"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.5,
        },
        "objaverse": {
            "scale": 1.5,
        },
    },
    "ketchup": {
        "types": ("condiment"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.35,
        },
        "objaverse": {
            "exclude": [
                "ketchup_5"  # causing error: faces of mesh have inconsistent orientation.
            ]
        },
    },
    "kettle_electric": {
        "types": ("receptacle"),
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "objaverse": {
            "scale": 1.35,
            "model_folders": ["objaverse/kettle"],
            "exclude": [
                f"kettle_{i}"
                for i in range(29)
                if i not in [0, 7, 9, 12, 13, 17, 24, 25, 26, 27]
            ],
        },
        "aigen": {
            "scale": 1.5,
            "model_folders": ["aigen_objs/kettle"],
            "exclude": [
                f"kettle_{i}" for i in range(11) if i not in [0, 2, 6, 9, 10, 11]
            ],
        },
    },
    "kettle_non_electric": {
        "types": ("receptacle"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "objaverse": {
            "scale": 1.35,
            "model_folders": ["objaverse/kettle"],
            "exclude": [
                f"kettle_{i}"
                for i in range(29)
                if i in [0, 7, 9, 12, 13, 17, 24, 25, 26, 27]
            ],
        },
        "aigen": {
            "scale": 1.5,
            "model_folders": ["aigen_objs/kettle"],
            "exclude": [f"kettle_{i}" for i in range(11) if i in [0, 2, 6, 9, 10, 11]],
        },
    },
    "kiwi": {
        "types": ("fruit"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": 0.90,
        },
        "objaverse": {
            "scale": 0.90,
        },
    },
    "knife": {
        "types": ("utensil"),
        "graspable": False,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": False,
        "aigen": {
            "scale": 1.35,
        },
        "objaverse": {
            "scale": 1.20,
        },
    },
    "ladle": {
        "types": ("utensil"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": True,
        "freezable": False,
        "aigen": {
            "scale": 1.5,
        },
        "objaverse": {
            "scale": 1.10,
        },
    },
    "lemon": {
        "types": ("vegetable"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": True,
        "freezable": True,
        "aigen": {
            "scale": 1.1,
        },
        "objaverse": {},
    },
    "lime": {
        "types": ("vegetable"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": True,
        "freezable": True,
        "objaverse": {
            "scale": 1.0,
        },
        "aigen": {
            "scale": 0.90,
        },
    },
    "mango": {
        "types": ("fruit"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": True,
        "freezable": True,
        "aigen": {
            "scale": 1.0,
        },
        "objaverse": {
            "scale": 0.85,
            "exclude": [
                "mango_3",  # one half is pitch dark
            ],
        },
    },
    "milk": {
        "types": ("dairy", "drink"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.35,
        },
        "objaverse": {
            "exclude": [
                "milk_6"  # causing error: eigenvalues of mesh inertia violate A + B >= C
            ]
        },
    },
    "mug": {
        "types": ("receptacle", "stackable"),
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.3,
        },
        "objaverse": {},
    },
    "mushroom": {
        "types": ("vegetable"),
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "aigen": {
            "scale": 1.35,
        },
        "objaverse": {
            "scale": 1.20,
            "exclude": [
                # "mushroom_16", # very very small holes. keeping anyway
            ],
        },
    },
    "onion": {
        "types": ("vegetable"),
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": False,
        "aigen": {
            "scale": 1.1,
        },
        "objaverse": {},
    },
    "orange": {
        "types": ("fruit"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.05,
        },
        "objaverse": {
            "exclude": [
                # "orange_11", # bottom half is dark. keeping anyway
            ]
        },
    },
    "pan": {
        "types": ("receptacle"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 2.25,
        },
        "objaverse": {
            "scale": 1.70,
            "exclude": [
                "pan_16",  # causing error. faces of mesh have inconsistent orientation,
                "pan_0",
                "pan_12",
                "pan_17",
                "pan_22",  # these are technically what we consider "pots"
            ],
        },
    },
    "pot": {
        "types": ("receptacle"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 2.25,
        },
        "objaverse": {
            "model_folders": ["objaverse/pan"],
            "scale": 1.70,
            "exclude": list(
                {f"pan_{i}" for i in range(25)}
                - {"pan_0", "pan_12", "pan_17", "pan_22"}
            ),
        },
    },
    "peach": {
        "types": ("fruit"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.05,
        },
        "objaverse": {},
    },
    "pear": {
        "types": ("fruit"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {},
        "objaverse": {
            "exclude": [
                "pear_4",  # has big hole. excluding
            ]
        },
    },
    "plate": {
        "types": ("receptacle"),
        "graspable": False,
        "washable": True,
        "microwavable": True,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.65,
        },
        "objaverse": {
            "scale": 1.35,
            "exclude": [
                "plate_6",  # causing error: faces of mesh have inconsistent orientation.
            ],
        },
    },
    "potato": {
        "types": ("vegetable"),
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "aigen": {
            "scale": 1.10,
        },
        "objaverse": {},
    },
    "rolling_pin": {
        "types": ("tool"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.6,
        },
        "objaverse": {
            "scale": 1.25,
            "exclude": [
                # "rolling_pin_5", # can see thru side handle edges, keeping anyway
                # "rolling_pin_1", # can see thru side handle edges, keeping anyway
            ],
        },
    },
    "scissors": {
        "types": ("tool"),
        "graspable": False,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.35,
        },
        "objaverse": {
            "scale": 1.15,
        },
    },
    "shaker": {
        "types": ("condiment"),
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.25,
        },
        "objaverse": {},
    },
    "soap_dispenser": {
        "types": ("cleaner"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.7,
        },
        "objaverse": {
            "exclude": [
                # "soap_dispenser_4", # can see thru body but that's fine if this is glass
            ]
        },
    },
    "spatula": {
        "types": ("utensil"),
        "graspable": False,
        "washable": True,
        "microwavable": False,
        "cookable": True,
        "freezable": False,
        "aigen": {
            "scale": 1.30,
        },
        "objaverse": {
            "scale": 1.10,
        },
    },
    "sponge": {
        "types": ("cleaner"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.20,
        },
        "objaverse": {
            "scale": 0.90,
            # exclude=[
            #     "sponge_7", "sponge_1", # self turning due to single collision geom
            # ]
        },
    },
    "spoon": {
        "types": ("utensil"),
        "graspable": False,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": False,
        "aigen": {
            "scale": 1.5,
        },
        "objaverse": {},
    },
    "spray": {
        "types": ("cleaner"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.75,
        },
        "objaverse": {
            "scale": 1.75,
        },
    },
    "squash": {
        "types": ("vegetable"),
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "aigen": {
            "scale": 1.15,
        },
        "objaverse": {
            "exclude": [
                "squash_10",  # hole at bottom
            ],
        },
    },
    "steak": {
        "types": ("meat"),
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "aigen": {
            "scale": [1.0, 1.0, 2.0],
        },
        "objaverse": {
            "scale": [1.0, 1.0, 2.0],
            "exclude": [
                "steak_13",  # bottom texture completely messed up
                "steak_1",  # bottom texture completely messed up
                # "steak_9", # bottom with some minor issues, keeping anyway
            ],
        },
    },
    "sweet_potato": {
        "types": ("vegetable"),
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "aigen": {},
        "objaverse": {},
    },
    "tangerine": {
        "types": ("fruit"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {},
        "objaverse": {},
    },
    "teapot": {
        "types": ("receptacle"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.25,
        },
        "objaverse": {
            "scale": 1.25,
            "exclude": [
                "teapot_9",  # hole on bottom
            ],
        },
    },
    "tomato": {
        "types": ("vegetable"),
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": False,
        "aigen": {
            "scale": 1.25,
        },
        "objaverse": {},
    },
    "tray": {
        "types": ("receptacle"),
        "graspable": False,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {"scale": 2.0},
        "objaverse": {
            "scale": 1.80,
        },
    },
    "waffle": {
        "types": ("sweets"),
        "graspable": False,
        "washable": False,
        "microwavable": True,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": 1.75,
        },
        "objaverse": {
            "exclude": [
                "waffle_2",  # bottom completely messed up
            ]
        },
    },
    "water_bottle": {
        "types": ("drink"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": 1.6,
        },
        "objaverse": {
            "scale": 1.5,
            "exclude": [
                "water_bottle_11",  # sides and bottom see thru, but ok if glass. keeping anyway
            ],
        },
    },
    "wine": {
        "types": ("drink", "alcohol"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "aigen": {
            "scale": 1.9,
        },
        "objaverse": {
            "scale": 1.6,
            "exclude": [
                "wine_7",  # causing error. faces of mesh have inconsistent orientation
            ],
        },
    },
    "yogurt": {
        "types": ("dairy", "packaged_food"),
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "aigen": {
            "scale": 1.0,
        },
        "objaverse": {
            "scale": 0.95,
        },
    },
    "dates": {
        "graspable": False,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "types": ("fruit"),
        "aigen": {},
    },
    "lemonade": {
        "aigen": {
            "scale": 1.5,
        },
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": ("drink"),
    },
    "walnut": {
        "aigen": {
            "scale": 1.15,
        },
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": (),
    },
    "cheese_grater": {
        "aigen": {
            "scale": 2.15,
        },
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": ("tool"),
    },
    "syrup_bottle": {
        "aigen": {
            "scale": 1.35,
        },
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": ("condiment"),
    },
    "scallops": {
        "aigen": {
            "scale": 1.25,
        },
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "types": ("meat"),
    },
    "candy": {
        "aigen": {},
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": ("sweets"),
    },
    "whisk": {
        "aigen": {
            "scale": 1.8,
        },
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": ("utensil"),
    },
    "pitcher": {
        "aigen": {
            "scale": 1.75,
        },
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": False,
        "freezable": False,
        "types": ("receptacle"),
    },
    "ice_cream": {
        "aigen": {
            "scale": 1.25,
        },
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "types": ("sweets"),
    },
    "cherry": {
        "aigen": {},
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "types": ("fruit"),
    },
    "peanut_butter": {
        "aigen": {
            "scale": 1.25,
            "model_folders": ["aigen_objs/peanut_butter_jar"],
        },
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "types": ("packaged_food"),
    },
    "thermos": {
        "aigen": {
            "scale": 1.75,
        },
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": False,
        "freezable": True,
        "types": ("drink"),
    },
    "ham": {
        "aigen": {
            "scale": 1.5,
        },
        "graspable": False,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "types": ("meat"),
    },
    "dumpling": {
        "aigen": {
            "scale": 1.15,
        },
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "types": ("meat", "cooked_food"),
    },
    "cabbage": {
        "aigen": {
            "scale": 2.0,
        },
        "graspable": False,
        "washable": True,
        "microwavable": False,
        "cookable": True,
        "freezable": True,
        "types": ("vegetable"),
    },
    "lettuce": {
        "aigen": {
            "scale": 2.0,
        },
        "graspable": False,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "types": ("vegetable"),
    },
    "tongs": {
        "aigen": {
            "scale": 1.5,
        },
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": ("tool"),
    },
    "ginger": {
        "aigen": {
            "scale": 1.35,
        },
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": True,
        "freezable": True,
        "types": ("vegetable"),
    },
    "ice_cube_tray": {
        "aigen": {
            "scale": 2.0,
        },
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "types": ("receptacle"),
    },
    "shrimp": {
        "aigen": {
            "scale": 1.15,
        },
        "graspable": False,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "types": ("meat"),
    },
    "cantaloupe": {
        "aigen": {
            "scale": 1.5,
        },
        "graspable": False,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "types": ("fruit"),
    },
    "honey_bottle": {
        "aigen": {
            "scale": 1.10,
        },
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "types": ("packaged_food"),
    },
    "grapes": {
        "aigen": {
            "scale": 1.5,
        },
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "types": ("fruit"),
    },
    "spaghetti_box": {
        "aigen": {
            "scale": 1.25,
        },
        "graspable": False,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": ("packaged_food"),
    },
    "chili_pepper": {
        "aigen": {
            "scale": 1.10,
        },
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "types": ("vegetable"),
    },
    "celery": {
        "aigen": {
            "scale": 2.0,
        },
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "types": ("vegetable"),
    },
    "burrito": {
        "aigen": {
            "scale": 1.35,
        },
        "graspable": True,
        "washable": False,
        "microwavable": True,
        "cookable": False,
        "freezable": True,
        "types": ("cooked_food"),
    },
    "olive_oil_bottle": {
        "aigen": {
            "scale": 1.5,
        },
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "types": ("packaged_food"),
    },
    "kebabs": {
        "aigen": {
            "scale": 1.65,
        },
        "graspable": True,
        "washable": False,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "types": ("cooked_food"),
    },
    "bottle_opener": {
        "aigen": {},
        "graspable": False,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "types": ("tool"),
    },
    "chicken_breast": {
        "aigen": {
            "scale": 1.35,
        },
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "types": ("meat"),
    },
    "jello_cup": {
        "aigen": {
            "scale": 1.15,
        },
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "types": ("packaged_food"),
    },
    "lobster": {
        "aigen": {
            "scale": 1.15,
        },
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "types": ("meat"),
    },
    "brussel_sprout": {
        "aigen": {},
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "types": ("vegetable"),
    },
    "sushi": {
        "aigen": {
            "scale": 0.90,
        },
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "types": ("meat"),
    },
    "baking_sheet": {
        "aigen": {
            "scale": 1.75,
        },
        "graspable": False,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": ("receptacle"),
    },
    "wine_glass": {
        "aigen": {
            "scale": 1.5,
        },
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": False,
        "freezable": True,
        "types": ("receptacle"),
    },
    "asparagus": {
        "aigen": {
            "scale": 1.35,
        },
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "types": ("vegetable"),
    },
    "lamb_chop": {
        "aigen": {
            "scale": 1.15,
        },
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "types": ("meat"),
    },
    "pickle": {
        "aigen": {
            "scale": 1.0,
        },
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "types": ("vegetable"),
    },
    "bacon": {
        "aigen": {
            "scale": 1.35,
        },
        "graspable": False,
        "washable": False,
        "microwavable": True,
        "cookable": True,
        "freezable": False,
        "types": ("meat"),
    },
    "canola_oil": {
        "aigen": {
            "scale": 1.75,
        },
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": ("packaged_food"),
    },
    "strawberry": {
        "aigen": {
            "scale": 0.9,
        },
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "types": ("fruit"),
    },
    "watermelon": {
        "aigen": {
            "scale": 2.5,
        },
        "graspable": False,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": ("fruit"),
    },
    "pizza_cutter": {
        "aigen": {
            "scale": 1.4,
        },
        "graspable": False,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": ("tool"),
    },
    "pomegranate": {
        "aigen": {
            "scale": 1.25,
        },
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": ("fruit"),
    },
    "apricot": {
        "aigen": {
            "scale": 0.7,
        },
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": ("fruit"),
    },
    "beet": {
        "aigen": {},
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": True,
        "freezable": False,
        "types": ("vegetable"),
    },
    "radish": {
        "aigen": {
            "scale": 1.0,
        },
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": ("vegetable"),
    },
    "salsa": {
        "aigen": {
            "scale": 1.15,
        },
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": ("packaged_food"),
    },
    "artichoke": {
        "aigen": {
            "scale": 1.35,
        },
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": True,
        "freezable": False,
        "types": ("vegetable"),
    },
    "scone": {
        "aigen": {
            "scale": 1.35,
        },
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": ("pastry", "bread_food"),
    },
    "hamburger": {
        "aigen": {
            "scale": 1.35,
        },
        "graspable": True,
        "washable": False,
        "microwavable": True,
        "cookable": False,
        "freezable": False,
        "types": ("cooked_food"),
    },
    "raspberry": {
        "aigen": {
            "scale": 0.85,
        },
        "graspable": False,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "types": ("fruit"),
    },
    "tacos": {
        "aigen": {
            "scale": 1.0,
        },
        "graspable": True,
        "washable": False,
        "microwavable": True,
        "cookable": False,
        "freezable": False,
        "types": ("cooked_food"),
    },
    "vinegar": {
        "aigen": {
            "scale": 1.4,
        },
        "graspable": True,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": ("packaged_food", "condiment"),
    },
    "zucchini": {
        "aigen": {
            "scale": 1.35,
        },
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "types": ("vegetable"),
    },
    "pork_loin": {
        "aigen": {},
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "types": ("meat"),
    },
    "pork_chop": {
        "aigen": {
            "scale": 1.25,
        },
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "types": ("meat"),
    },
    "sausage": {
        "aigen": {
            "scale": 1.45,
        },
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "types": ("meat"),
    },
    "coconut": {
        "aigen": {
            "scale": 2.0,
        },
        "graspable": False,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": ("fruit"),
    },
    "cauliflower": {
        "aigen": {
            "scale": 1.5,
        },
        "graspable": False,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "types": ("vegetable"),
    },
    "lollipop": {
        "aigen": {},
        "graspable": False,
        "washable": False,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": ("sweets"),
    },
    "salami": {
        "aigen": {
            "scale": 1.5,
        },
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "types": ("meat"),
    },
    "butter_stick": {
        "aigen": {
            "scale": 1.3,
        },
        "graspable": True,
        "washable": False,
        "microwavable": True,
        "cookable": True,
        "freezable": True,
        "types": ("dairy"),
    },
    "can_opener": {
        "aigen": {
            "scale": 1.5,
        },
        "graspable": False,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": False,
        "types": ("tool"),
    },
    "tofu": {
        "aigen": {},
        "graspable": True,
        "washable": True,
        "microwavable": False,
        "cookable": True,
        "freezable": True,
        "types": (),
    },
    "pineapple": {
        "aigen": {
            "scale": 2.0,
        },
        "graspable": False,
        "washable": True,
        "microwavable": False,
        "cookable": False,
        "freezable": True,
        "types": ("fruit"),
    },
    "skewers": {
        "aigen": {
            "scale": 1.75,
        },
        "graspable": True,
        "washable": True,
        "microwavable": True,
        "cookable": True,
        "freezable": False,
        "types": ("meat", "cooked_food"),
    },
}


def get_cats_by_type(types, obj_registries=None):
    """Retrieves a list of item keys from the global `OBJ_CATEGORIES` dictionary based on the specified types.

    Args:
        types (list): A list of valid types to filter items by. Only items with a matching type will be included.
        obj_registries (list): only consider categories belonging to these object registries

    Returns:
        list: A list of keys from `OBJ_CATEGORIES` where the item's types intersect with the provided `types`.

    """
    types = set(types)

    res = []
    for key, val in OBJ_CATEGORIES.items():
        # check if category is in one of valid object registries
        if obj_registries is not None:
            if isinstance(obj_registries, str):
                obj_registries = [obj_registries]
            if any(reg in val for reg in obj_registries) is False:
                continue

        cat_types = val["types"] if "types" in val else list(val.values())[0].types
        if isinstance(cat_types, str):
            cat_types = [cat_types]
        cat_types = set(cat_types)
        # Access the "types" key in the dictionary using the correct syntax
        if len(cat_types.intersection(types)) > 0:
            res.append(key)

    return res


### define all object categories ###
OBJ_GROUPS = {
    "all": list(OBJ_CATEGORIES.keys()),
}

for k in OBJ_CATEGORIES:
    OBJ_GROUPS[k] = [k]

all_types = set()
# populate all_types
for _cat, cat_meta_dict in OBJ_CATEGORIES.items():
    # types are common to both so we only need to examine one
    cat_types = cat_meta_dict["types"]
    if isinstance(cat_types, str):
        cat_types = [cat_types]
    all_types = all_types.union(cat_types)

for t in all_types:
    OBJ_GROUPS[t] = get_cats_by_type(types=[t])

OBJ_GROUPS["food"] = get_cats_by_type(
    [
        "vegetable",
        "fruit",
        "sweets",
        "dairy",
        "meat",
        "bread_food",
        "pastry",
        "cooked_food",
    ]
)
OBJ_GROUPS["in_container"] = get_cats_by_type(
    [
        "vegetable",
        "fruit",
        "sweets",
        "dairy",
        "meat",
        "bread_food",
        "pastry",
        "cooked_food",
    ]
)

# custom groups
OBJ_GROUPS["container"] = ["plate"]  # , "bowl"]
OBJ_GROUPS["kettle"] = ["kettle_electric", "kettle_non_electric"]
OBJ_GROUPS["cookware"] = ["pan", "pot", "kettle_non_electric"]
OBJ_GROUPS["pots_and_pans"] = ["pan", "pot"]
OBJ_GROUPS["food_set1"] = [
    "apple",
    "baguette",
    "banana",
    "carrot",
    "cheese",
    "cucumber",
    "egg",
    "lemon",
    "orange",
    "potato",
]
OBJ_GROUPS["group1"] = ["apple", "carrot", "banana", "bowl", "can"]
OBJ_GROUPS["container_set2"] = ["plate", "bowl"]
