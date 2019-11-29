$(document).ready(function(){
   $("#submitReport").on("click", function(e){
       e.preventDefault();
       $("#reportform").hide();
       $(".reportText").hide();
       $(".greetings").show();
   });
});