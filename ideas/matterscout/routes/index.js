var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'MS' });
});

router.get('/report', function(req, res, next) {
  res.render('report', { title: 'MS' });
});
module.exports = router;
