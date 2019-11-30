$(document).ready(function () {
    $("#tagbutton").on("click",function (e) {
        e.preventDefault();
        let inserted_tag = $('#tag-form').val();
        let tags = $('#labels').html();

        new_tag = '<span class="uk-label">'+inserted_tag+'</span>';

        $('#labels').html(tags+new_tag);
        $('#tag-form').val("");

    })

})