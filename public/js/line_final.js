function createoption(obj,SelectedNodeID,startIndex = -1,endIndex = -1, methodid){
        
    var xdata = new Array();

    // 处理日期作为x轴
    const Gap = 1000*60*now_Timefitness; // 将timefitness转化为ms
    var EndDate = +TimeRange[2]
    console.log("EndDate:", EndDate)
    console.log(TimeRange[2])

    var DatasetName = DatasetList[DatasetID];
    var tmp = obj.Pred[DatasetName];
    var TimeNum = tmp.GroundTruth.length;

    // 默认datazoom位置

    if(startIndex > TimeNum | endIndex > TimeNum | startIndex < -1 | endIndex < -1)
    {
        window.alert("Error 02: out of range!");
        startIndex = -1;
        endIndex = -1;
    }

    if(startIndex == -1)
        startIndex = Math.floor(TimeNum*0.9);
    if(endIndex == -1)
        endIndex = TimeNum - 1;

    if(startIndex > endIndex)
    {
        window.alert("Error 01: wrong time selected");
        startIndex = -1;
        endIndex = -1;
    }

    // 获得x轴的刻度
    for (i = TimeNum - 1; i >= 0; i--) {
        now = new Date(EndDate - (TimeNum - i)*Gap);
        xdata[i] = [now.getFullYear(),now.getMonth()+1,now.getDate()].join('/') +' '+[now.getHours(),now.getMinutes()].join(':');
    }

    // ydata：这个数据点各个时间的真实值 获得真实值数组
    var RealNodeID = -1;  // 记录这个节点对应在traffic_data_index里的位置，-1代表没这个玩意
    var methodName = MethodNameArray[methodid];
    console.log("now_methodName:", methodName);
    RealNodeID = tmp[methodName].traffic_data_index.indexOf(SelectedNodeID);
    var ydata = new Array();
    for(i = 0; i<TimeNum;i++) {
        ydata[i] = tmp.GroundTruth[i][RealNodeID];
    }

    // pred_data: 预测值
    var pred_data = new Array();
    // 如果存在这个点,把预测值填入predata数组（采用尾端对齐）
    if(RealNodeID != -1) {
            // 遍历所有时间点
        for (k = 0; k < tmp[methodName].TrafficNode.length; k++){     // 真实值和预测值的时间长度不等，预测的段应该是尾端对齐
            var index = tmp[methodName].TrafficNode.length-k-1
            pred_data[TimeNum-1-k] = tmp[methodName].TrafficNode[index][RealNodeID];
        }
    }

    var color_list = ['#2f4554', '#61a0a8', '#FF8C00', '#20B2AA','#ffc20e','#4169E1','#afb4db', '#c4ccd3'];
    var j = 0;

/* 注：这里有个问题
    因为拿到的数据集特殊，都是0.1划分，所以程序是按0.1写的
    但实际上不一定是0.1，所以Pred里头数据集才附带了真实值。
    现在并没有使用这个真实值，但可能会有问题。
*/

    // 指定图表的配置项和数据
    var option = {
        legend: {orient:'horizontal',left:'2%', top:'2%', textStyle:{color:'#fff'}},
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'cross',
            },
            formatter: function (params, ticket, callback) {

                var result = params[0].axisValue + '<br/>';

                result += '<span style="display:inline-block;position:relative; top:-3px;margin-right:5px;border-radius:10px;width:9px;height:9px;background-color:' + params[0].color +'"></span>'
                           + params[0].seriesName + ' : ' + params[0].data.toFixed(2) + '<br/>' ;

                for(i = 1; i< params.length;i++)
                {
                    /* 结果 */
                    result += '<span style="display:inline-block;position:relative; top:-3px;margin-right:5px;border-radius:10px;width:9px;height:9px;background-color:' + params[i].color +'"></span>'
                           + params[i].seriesName + ' : ' + params[i].data.toFixed(2) + '<br/>';

                    /* 相对误差 */
                    result += '&nbsp&nbsp&nbsp&nbsp' +'<span style="display:inline-block;position:relative; top:-3px;margin-right:5px;border-radius:6px;width:5px;height:5px;background-color:' + params[i].color +'"></span>'
                           + 'RE: ';

                    var relative_error = Math.abs((params[i].data-params[0].data)/params[0].data);
                    if(params[0].data == 0)
                    {
                        if(params[1].data == 0){error_rate = 0;}
                        result +=  (relative_error*100).toFixed(2) + '&nbsp;';
                    }
                    else
                    {
                        result += (relative_error*100).toFixed(2) + '%&nbsp;';
                    }

                    /* 绝对误差 */
                    result += '&nbsp&nbsp&nbsp' +'<span style="display:inline-block;position:relative; top:-3px;margin-right:5px;border-radius:6px;width:5px;height:5px;background-color:' + params[i].color +'"></span>'
                    + 'AE: ';

                    var absolute_error = Math.abs(params[i].data-params[0].data);
                    result += absolute_error.toFixed(2) + '<br/>';

                    /* 判断结果 */
                    result += '&nbsp&nbsp&nbsp&nbsp' + '<span style="display:inline-block;position:relative; top:-3px;margin-right:5px;border-radius:6px;width:5px;height:5px;background-color:' + params[i].color +'"></span>'

                    if(relative_error <= MAXError)
                    {
                        if(absolute_error <= MAXABSError)
                        {
                            result += 'Result: ' + '√' + '<br/>';
                        }
                        else
                        {
                            // 暂时还是没有想到太合适的方法把AE纳入评价依据，暂时不用
                            // result += 'Result: ' + '× (AE)' + '<br/>';
                            result += 'Result: ' + '√' + '<br/>';
                        }
                    }
                    else if(absolute_error <= MAXABSError)
                    {
                        result += 'Result: ' + '× (RE)' + '<br/>';
                    }
                    else
                    {
                        result += 'Result: ' + '× (Both)' + '<br/>';
                    }
                }

                return [result];
            },

            position: function (pt) {
                return [pt[0], '10%'];
            }
        },
        color:['#2f4554', '#61a0a8', '#d48265', '#bda29a','#6e7074','#87CEFA','#546570', '#c4ccd3'],
        axisPointer: {
            label: {
                backgroundColor: '#1177',
                precision:2
            }
        },
        xAxis: {
            type: 'category',
            data: xdata
        },
        yAxis: {
            type: 'value'
        },
        series: function() {
            var Myseries = new Array();
            var item =
            {
                name: "Ground Truth",
                data: ydata,
                type: 'line',
                symbol:'triangle',
                symbolSize: 8,
                itemStyle: {
                    borderColor: "#111AAA",
                    color: '#fff',
                    shadowColor: 'rgba(0, 0, 0, 0.5)',
                    shadowBlur: 10
                },
                lineStyle: {
                    width : 3
                },
            }
            Myseries.push(item);

            var item = {
                name: methodName,
                data: pred_data,
                type: 'line',
                color: color_list[j+2],
                symbol: function(number,params) {
                    var index = params.dataIndex;
                    var re = Math.abs((number - ydata[index])) / ydata[index];
                    if(ydata[index] == 0) {
                        return 'emptyCircle';
                    }
                    if(line_highlight == 1) {
                        if(re <= MAXError) {
                            return 'image://./images/greendot2.png';
                        }
                        else {
                            return 'image://./images/reddot.png';
                        }
                    } else {
                        return 'emptyCircle';
                    }
                },
                //showSymbol : false,
                showAllSymbol : true,
                symbolSize: function(number,params) {
                    var index = params.dataIndex;
                    var re = Math.abs((number - ydata[index])) / ydata[index];

                    if(ydata[index] == 0) {
                        return 2;
                    }

                    if(line_highlight == 1) {
                        if(re <= MAXError) {
                            return 8;
                        }
                        else {
                            return 8;
                        }
                    }
                    else {
                        return 4;
                    }
                },
            }
            Myseries.push(item);

            return Myseries;
        }(),
        dataZoom: [
        {
                type: 'slider',
                show: true,
                xAxisIndex: 0,
                //filterMode: 'empty',   //这句话加上的话，不会随着数据改变轴
                startValue: xdata[startIndex],
                endValue: xdata[endIndex],
                handleIcon: 'M10.7,11.9v-1.3H9.3v1.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4v1.3h1.3v-1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
                handleSize: '80%',
                handleStyle: {
                    color: '#fff',
                    shadowBlur: 3,
                    shadowColor: 'rgba(0, 0, 0, 0.6)',
                    shadowOffsetX: 2,
                    shadowOffsetY: 2
                }
        },
        {
            type: 'inside',
            show: true,
            xAxisIndex: 0,
            startValue: xdata[startIndex],
            endValue: xdata[endIndex],
        },
        {
            type: 'slider',
            yAxisIndex: 0,
            handleIcon: 'M10.7,11.9v-1.3H9.3v1.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4v1.3h1.3v-1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
            handleSize: '80%',
            handleStyle: {
                color: '#fff',
                shadowBlur: 3,
                shadowColor: 'rgba(0, 0, 0, 0.6)',
                shadowOffsetX: 2,
                shadowOffsetY: 2
            }
        }
        ]
    };
    // 使用刚指定的配置项和数据显示图表。

    return option;
}

function drawline(option){
    
    var myChart = echarts.init(document.getElementById("container_line"));
    myChart.clear();
    myChart.setOption(option);
    myChart.on('datazoom', function (params) {

        let startValue = myChart.getOption().dataZoom[1].startValue;
        let endValue = myChart.getOption().dataZoom[1].endValue;

        const Gap = 1000*60*now_Timefitness;
        var Origin_endDate = TimeRange[2];

        var now = new Date(Origin_endDate-Gap*(TimeArrayLength-startValue));
        var nowend = new Date(Origin_endDate-Gap*(TimeArrayLength-endValue));

        var start_year = now.getFullYear().toString();
        var start_month = (now.getMonth()+1).toString();
        var start_day = now.getDate().toString();
        var start_hour = now.getHours().toString();
        var start_minute = now.getMinutes().toString();
       
        if(now.getMonth()+1 < 10){start_month = '0' + start_month;}
        if(now.getDate() < 10){start_day = '0' + start_day;}
        if(now.getHours() < 10){start_hour = '0' + start_hour;}
        if(now.getMinutes() < 10){start_minute = '0' + start_minute;}

        var now_time = start_year + '-' + start_month + '-' + start_day + 'T' + start_hour + ':' + start_minute + ':00';
        document.getElementById('starttime').value = now_time;

        var end_year = nowend.getFullYear().toString();
        var end_month = (nowend.getMonth()+1).toString();
        var end_day = nowend.getDate().toString();
        var end_hour = nowend.getHours().toString();
        var end_minute = nowend.getMinutes().toString();
       
        if(nowend.getMonth()+1 < 10){end_month = '0' + end_month;}
        if(nowend.getDate() < 10){end_day = '0' + end_day;}
        if(nowend.getHours() < 10){end_hour = '0' + end_hour;}
        if(nowend.getMinutes() < 10){end_minute = '0' + end_minute;}

        var nowend_time = end_year + '-' + end_month + '-' + end_day + 'T' + end_hour + ':' + end_minute + ':00';
        document.getElementById('endtime').value = nowend_time;

        StartInd = startValue;
        EndInd = endValue;
    });

}

function createRMSEoption(methodid, pointid){

    var xdata = new Array();
    var ydata = new Array();
    var PointsList = PointSortedVar[methodid];

    for(var i=0; i<PointsList.length; i++) {
        dict = PointsList[i];
        // 获得x轴数据
        xdata.push(dict['index']);
        // 获得y轴数据
        ydata.push(dict['rmse']);
    }

    option = {
        legend : {
            left: '7%',
            textStyle:{color:'#fff'}
        },
        xAxis: {
            type: 'category',
            data: xdata
        },
        yAxis: {
            type: 'value',
        },
        tooltip: {
            show: true,
            trigger: 'axis'
        },
        series: {
                data: function() {
                    var MyData = new Array();
                    if(line_highlight != 1){
                        for(i=0; i<ydata.length; i++){
                            if(i != pointid){
                                MyData.push(ydata[i]);
                            }
                            else {
                                item = {
                                    value: ydata[i],
                                    itemStyle: {
                                        color: '#FFD700'
                                    }
                                }
                                MyData.push(item);
                            }
                        }
                    }
                    else {
                        for(i=0; i<ydata.length; i++){
                            if(ydata[i] < MINACCURACY){
                                MyData.push(ydata[i]);
                            }
                            else {
                                item = {
                                    value: ydata[i],
                                    itemStyle: {
                                        color: '#FFE4C4'
                                    }
                                }
                                MyData.push(item);
                            }
                        }
                    }
                    return MyData;
                }(),
                type: 'bar',
                name: 'RMSE',
                itemStyle: {
                    normal: {
                        color: '#d48265'
                    }
                },
        },
        dataZoom: [
            {
                type: 'slider',
                show: true,
                xAxisIndex: 0,
                //filterMode: 'empty',   //这句话加上的话，不会随着数据改变轴
                startValue: 0,
                endValue: 30,
                handleIcon: 'M10.7,11.9v-1.3H9.3v1.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4v1.3h1.3v-1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
                handleSize: '80%',
                handleStyle: {
                    color: '#fff',
                    shadowBlur: 3,
                    shadowColor: 'rgba(0, 0, 0, 0.6)',
                    shadowOffsetX: 2,
                    shadowOffsetY: 2
                }
            },
            {
                type: 'inside',
                show: true,
                xAxisIndex: 0,
            },
            {
                type: 'slider',
                yAxisIndex: 0,
                handleIcon: 'M10.7,11.9v-1.3H9.3v1.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4v1.3h1.3v-1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
                handleSize: '80%',
                handleStyle: {
                    color: '#fff',
                    shadowBlur: 3,
                    shadowColor: 'rgba(0, 0, 0, 0.6)',
                    shadowOffsetX: 2,
                    shadowOffsetY: 2
                }
            }
        ]
    };

    return option;
}

function drawRMSEhistogram(option){
    var myChart = echarts.init(document.getElementById('rmseline2'));
    myChart.setOption(option);
    window.addEventListener("resize",function(){
        myChart.resize();
    });
}
