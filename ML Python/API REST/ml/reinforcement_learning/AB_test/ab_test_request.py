ab_test_request = {
    "type" : "object",
    "properties": {
        "real_chance_1" : {"type" : "number"},
        "real_chance_2" : {"type" : "number"},
        "real_chance_3" : {"type" : "number"},
        "events" : {"type" : "number"}
    },
    "required" : ["real_chance_1", "real_chance_2", "real_chance_3", "events"]
}