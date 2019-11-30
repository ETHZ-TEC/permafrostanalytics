Plotly.d3.csv("/data/2017-02-02_07:00:00/seismic_data.csv", function(err, rows){

    function unpack(rows, key) {
        return rows.map(function(row) { return row[key]; });
    }


    var trace1 = {
        type: "scatter",
        mode: "lines",
        name: 'Measurements',
        x: unpack(rows, 'date'),
        y: unpack(rows, 'EHE'),
        line: {color: '#F44336'}
    };

    var trace2 = {
        type: "scatter",
        mode: "lines",
        name: 'Median',
        x: unpack(rows, 'date'),

        y: unpack(rows, 'EHE-n'),
        line: {color: '#7F7F7F'}
    };

    var data = [trace1,trace2];

    var layout = {
        title: 'Seismic Data',
    };

    Plotly.newPlot('plot_data', data, layout, {showSendToCloud: true});
});