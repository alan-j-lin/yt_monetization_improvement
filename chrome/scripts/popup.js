$('#spinner').hide();

$('#autofill').click( () => {
  grab_url();
} );

$('#model').click( () => {
  categorize();
})


function grab_url(){

  chrome.tabs.query({
    'active': true,
    'currentWindow': true
  }, function(tabs) {
      tab_url = tabs[0].url;
      console.log(tab_url)
  });

  setTimeout(function(){$('#taburl').val(window.tab_url)}, 50);

};

function categorize(){

  // url that will be evaluated
  let features = {
    'url': $('#taburl').val(),
  };

  console.log(features);

  $('#results').hide();
  $('#spinner').fadeIn(1000);


  $.ajax({
    type: "POST",
    contentType: "application/json; charset=utf-8",
    url: "http://0.0.0.0:5000/classify",  // Replace with URL of POST handler
    dataType: "json",
    async: true,
    data: JSON.stringify(features),
    success: (result) => {

      let label = result['label']
      let probability = result['probability']
      
      $('#label').html(label)
      $('#probability').html(probability)

      $('#spinner').fadeOut(2000);
      $('#results').fadeIn(1000);

    },
    error: (result) => {
      alert('Everything is broken!');
    }
  })



};
