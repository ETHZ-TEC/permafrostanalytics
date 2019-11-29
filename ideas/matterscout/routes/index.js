var express = require('express');
var router = express.Router();

var result_data = {
  "1" : {
    "name": "Bastard avalanche",
    "description":"Oh man, fucking run away",
    "images": [1,2,3,4,5],
    "sensors": "1.csv",
    "time":"29-08-2019"
  },
  "2" : {
    "name": "Bastard avalanche",
    "description":"Oh man, fucking run away",
    "images": [1,2,3,4,5],
    "sensors": "1.csv",
    "time":"29-08-2019"
  }
};

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'MS' });
});


router.get('/event_detail/:id', function(req, res, next) {
  var event_id = req.params.id;
  res.render('event', result_data[event_id]);
});

router.get('/report', function(req, res, next) {
  res.render('report', { title: 'MS' });
});
module.exports = router;
