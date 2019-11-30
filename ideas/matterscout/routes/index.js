var express = require('express');
var router = express.Router();

var result_data = {
  1 : {
    "name": "Bastard avalanche",
    "description":"An avalanche fell down the mountain and killed everyone. A lot of snow has been detected, many mountaineers has been severely injured. Below you can see the pictures of the fact.",
    "images": [1,2,3,4,5],
    "sensors": "1.csv",
    "time":"29-08-2019",
    "id":1
  },
  2 : {
    "name": "Bastard avalanche",
    "description":"Oh man, fucking run away",
    "images": [1,2,3,4,5],
    "sensors": "1.csv",
    "time":"29-08-2019",
    "id":2
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
