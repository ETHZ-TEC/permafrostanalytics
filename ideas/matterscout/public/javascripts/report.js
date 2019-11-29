$(document).ready(function(){
   $("#submitReport").on("click", function(e){
       e.preventDefault();
       $("#reportcard").hide();
       $(".greetings").show();
   });
});