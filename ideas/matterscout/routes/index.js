var express = require('express');
var router = express.Router();

var result_data = {
  1 : {
    "name": "Heavy rain",
    "description":"We recorded an intense amount of rain, unusual for the period. Below you can see the details of the amount of rain recorded by our sensors.",
    "images": [1,2,3,4,5],
    "sensors": "1.csv",
    "time":"2017-02-02 07:00:00",
    "id":1,
    "severity":2
  },
  2 : {
    "name": "Hailstorm",
    "description":"We recorded a light hailstorm during the morning.",
    "images": [1,2,3,4,5],
    "sensors": "1.csv",
    "time":"2017-30-03-09:00:00",
    "id":2,
    "severity":1
  }
};

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('home', {data: result_data});
});

router.get('/events', function(req, res, next) {
  res.render('index', {data: result_data});
});

router.get('/report', function(req, res, next) {
  res.render('report');
});

router.get('/event_detail/:id', function(req, res, next) {
  const event_id = req.params.id;
  res.render('event_detail', result_data[event_id]);
});

router.get('/event_summary', function(req, res, next) {
  const event_id = req.params.id;
  res.send(result_data);
});

router.post('/report', function(req, res, next) {
  var json = JSON.stringify(req.body)
  var fs = require('fs');
  fs.writeFile("json/" + req.body.title + ".json", json, 'utf8', () => {console.log("file received")});
  res.render('greetings');
});

module.exports = router;
