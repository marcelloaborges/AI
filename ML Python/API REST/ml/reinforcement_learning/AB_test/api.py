from flask.views import MethodView
from flask import jsonify, request

import json
from jsonschema import Draft4Validator
from jsonschema.exceptions import best_match

from ml.reinforcement_learning.AB_test.ab_test_request import ab_test_request

from core.ml.reinforcement_learning.AB_test.upper_confidence_bound import UpperConfidenceBound

class ABTestApi(MethodView):

    def get(self):
        return "A/B Test Home"

    def post(self):
        ab_test_json = request.json
        error = best_match(Draft4Validator(ab_test_request).iter_errors(ab_test_json))
        
        if error:
            return jsonify({"error": error.message}), 400
        else:
            rc1 = ab_test_json.get('real_chance_1')
            rc2 = ab_test_json.get('real_chance_2')
            rc3 = ab_test_json.get('real_chance_3')
            events = ab_test_json.get('events')

            ucb = UpperConfidenceBound()
            result = ucb.Run(rc1, rc2, rc3, events)

            return jsonify(result), 200        