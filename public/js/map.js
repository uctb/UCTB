function drawmap(option){

    var dom = document.getElementById("map_1");
    var myChart = echarts.init(dom);

    if (option && typeof option === "object") {
        myChart.setOption(option, true);
    }

    myChart.on('click',function(params){
        pointID = params.dataIndex;
        pointName = params.value[3]
        document.getElementById('points').value = pointID;
        document.getElementById('uv_name').innerText = pointName;
        document.getElementById('line_graph').innerText = 'Groundtruth and Prediction (' + pointName + ')'

        /*右上折线图*/
        Newoption = createoption(data,pointID,StartInd,EndInd,MethodID);
        drawline(Newoption);

        /*error折线图*/
        RMSE_option = createRMSEoption(MethodID,pointID);
        drawRMSEhistogram(RMSE_option);

    });
}

function createMapOption(obj, centerID = -1,mapsize = myMapSize)
{
    option = null;
    console.log("obj:", obj)

    // 站点的数量
    var NodeNum = obj.Node.StationInfo.length;
    TotalPointNum = NodeNum;

    // 生成站点名称：坐标数据对
    var resData = [];
    var totallongitude = 0;
    var totallatitude = 0;
    for (var i = 0; i < NodeNum; i++) {
        if(typeof(obj.Node.StationInfo[i][3]) == "string")
        {
            totallongitude += parseFloat(obj.Node.StationInfo[i][3]);
            totallatitude += parseFloat(obj.Node.StationInfo[i][2]);
            resData.push({
                name: obj.Node.StationInfo[i][0],
                value: [parseFloat(obj.Node.StationInfo[i][3]),parseFloat(obj.Node.StationInfo[i][2]),i,obj.Node.StationInfo[i][4]]
            });
        }
        else
        {
            totallongitude += obj.Node.StationInfo[i][3];
            totallatitude += obj.Node.StationInfo[i][2];
            resData.push({
                name: obj.Node.StationInfo[i][0],                 // name属性是stationinfo的第一位（编号）
                value: [obj.Node.StationInfo[i][3],obj.Node.StationInfo[i][2],i,obj.Node.StationInfo[i][4]]   // value 包含坐标，和地点名称
            });
        }
    }
    console.log("resData:", resData)
    console.log("totallongitude/NodeNum", totallongitude/NodeNum);
    console.log(totallatitude/NodeNum);

    option = {
        tooltip : {
            trigger: 'item',
            formatter: function(params) {
                var res = "Point Node(first dimension of StationInfo): " + params.name+'<br/>';
                res += "Coordinates: ["+ params.data.value[0]+', '+ params.data.value[1]+'] </br>';
                res += "NodeName: "+ params.data.value[3]+'<br/>';
                res += "NodeID in dataset: " + params.data.value[2]+'<br/>';

                return [res];
            }
        },
        bmap: {
            center:[totallongitude/NodeNum, totallatitude/NodeNum],
            zoom: mapsize,
            roam: true,
            mapStyle: {
                styleJson: [{
                    'featureType': 'water',
                    'elementType': 'all',
                    'stylers': {
                        'color': '#d1d1d1'
                    }
                }, {
                    'featureType': 'land',
                    'elementType': 'all',
                    'stylers': {
                        'color': '#f3f3f3'
                    }
                }, {
                    'featureType': 'railway',
                    'elementType': 'all',
                    'stylers': {
                        'visibility': 'off'
                    }
                }, {
                    'featureType': 'highway',
                    'elementType': 'all',
                    'stylers': {
                        'color': '#fdfdfd'
                    }
                }, {
                    'featureType': 'highway',
                    'elementType': 'labels',
                    'stylers': {
                        'visibility': 'off'
                    }
                }, {
                    'featureType': 'arterial',
                    'elementType': 'geometry',
                    'stylers': {
                        'color': '#fefefe'
                    }
                }, {
                    'featureType': 'arterial',
                    'elementType': 'geometry.fill',
                    'stylers': {
                        'color': '#fefefe'
                    }
                }, {
                    'featureType': 'poi',
                    'elementType': 'all',
                    'stylers': {
                        'visibility': 'off'
                    }
                }, {
                    'featureType': 'green',
                    'elementType': 'all',
                    'stylers': {
                        'visibility': 'off'
                    }
                }, {
                    'featureType': 'subway',
                    'elementType': 'all',
                    'stylers': {
                        'visibility': 'off'
                    }
                }, {
                    'featureType': 'manmade',
                    'elementType': 'all',
                    'stylers': {
                        'color': '#d1d1d1'
                    }
                }, {
                    'featureType': 'local',
                    'elementType': 'all',
                    'stylers': {
                        'color': '#d1d1d1'
                    }
                }, {
                    'featureType': 'arterial',
                    'elementType': 'labels',
                    'stylers': {
                        'visibility': 'off'
                    }
                }, {
                    'featureType': 'boundary',
                    'elementType': 'all',
                    'stylers': {
                        'color': '#fefefe'
                    }
                }, {
                    'featureType': 'building',
                    'elementType': 'all',
                    'stylers': {
                        'color': '#d1d1d1'
                    }
                }, {
                    'featureType': 'label',
                    'elementType': 'labels.text.fill',
                    'stylers': {
                        'color': '#999999'
                    }
                }]
            }
        },
        series : [
            {
                name: 'Points',
                type: 'scatter',
                coordinateSystem: 'bmap',
                data: resData,
                symbolSize:10,
                itemStyle: {
                    color: function(params) {

                        var id = params.data.value[2];
                        var real_id = FindRealNodeID(id);

                        if(real_id == -1){return 'black';}
                        else if(PointMinVar[real_id] >= MINACCURACY){return 'red';}
                        else {return 'green';}
                    }
                },
                label: {
                    formatter: '{@[2]}',
                    position: 'right',
                    show: true
                },
                emphasis: {
                    label: {
                        show: false
                    }
                }
            },
        ]
    }

    /* 人为修改中心点 */
    if(centerID >= NodeNum)
    {
        window.alert("Map error 02 - pointID out of range!");
    }
    else if(centerID != -1)
    {
        var Myseries = option.series;
        var Value = Myseries[0].data[centerID].value;

        option.bmap.center = [Value[0],Value[1]];
    }

    /* 人为修改地图点的颜色 */

    return option;
}

function renderItem(params, api){
    // console.log(api.value(2))
    var coords = polygon[api.value(0)] //取出每一个区域的id
    var points = [];
    for (var i = 0; i < coords.length; i++) {
        points.push(api.coord(coords[i])); //points数组存放该区域边界的点
    }
    // console.log("points:", points)
    var color = api.visual('color'); //得到视觉映射的样式信息

    return {
        type: 'polygon',
        shape: {
            points: echarts.graphic.clipPointsByRect(points, {
                x: params.coordSys.x,
                y: params.coordSys.y,
                width: params.coordSys.width,
                height: params.coordSys.height
            })
        },
        style: api.style({
            fill: color,
            stroke: echarts.color.lift(color)
        })
    };
}

function map(){
    // 基于准备好的dom，初始化echarts实例
    var myChart = echarts.init(document.getElementById('map_1'));

    var NodeNum = data.Node.StationInfo.length;
    TotalPointNum = NodeNum;
    var resData = [];
    for(var i=0; i<VaildPointNum; i++){
        id = data.Node.StationInfo[i][4]
        resData.push({
            name: ConvertNameType[id][0],
            value: [i, ConvertNameType[id][1], data.Node.StationInfo[i][4]]
        });
    }

    option = {
        title: {
            left: 'center',
            top: '10px',
            textStyle: {
                color: '#fff'
            }
        },
        // tooltip : {
        //     trigger: 'item',
        //     formatter: function(params) {
        //         var res = params.name+'<br/>'+ '总流量:' + params.value[4];
        //         return res;
        //     }
        // },
        visualMap: {
            dimension: 1,
            categories: ['交通站点', '医院', '学校', '城中村', '展馆', '景点', '酒店', '商圈', '寺庙'],
            right: '3%',
            bottom: '3%',
            calculable: true,
            hoverLink: true,
            inRange: {
                color: ['#00FFFF', '#90EE90', '#FFD700', '#D2B48C', '#FF00FF', '#cde6c7', '#f8aba6', '#f58220', '#9b95c9']
            },
            textStyle: {
                color: '#fff'
            }
        },
        bmap: {
            center: [118.14363, 24.55285],
            zoom: 12,
            roam: true,
            mapStyle: {
                styleJson: [
                    {
                        "featureType": "water",
                        "elementType": "all",
                        "stylers": {
                            "color": "#044161"
                        }
                    },
                    {
                        "featureType": "land",
                        "elementType": "all",
                        "stylers": {
                            "color": "#004981"
                        }
                    },
                    {
                        "featureType": "boundary",
                        "elementType": "geometry",
                        "stylers": {
                            "color": "#064f85"
                        }
                    },
                    {
                        "featureType": "railway",
                        "elementType": "all",
                        "stylers": {
                            "visibility": "off"
                        }
                    },
                    {
                        "featureType": "highway",
                        "elementType": "geometry",
                        "stylers": {
                            "color": "#004981"
                        }
                    },
                    {
                        "featureType": "highway",
                        "elementType": "geometry.fill",
                        "stylers": {
                            "color": "#005b96",
                            "lightness": 1
                        }
                    },
                    {
                        "featureType": "highway",
                        "elementType": "labels",
                        "stylers": {
                            "visibility": "off"
                        }
                    },
                    {
                        "featureType": "arterial",
                        "elementType": "geometry",
                        "stylers": {
                            "color": "#004981"
                        }
                    },
                    {
                        "featureType": "arterial",
                        "elementType": "geometry.fill",
                        "stylers": {
                            "color": "#00508b"
                        }
                    },
                    {
                        "featureType": "poi",
                        "elementType": "all",
                        "stylers": {
                            "visibility": "off"
                        }
                    },
                    {
                        "featureType": "green",
                        "elementType": "all",
                        "stylers": {
                            "color": "#056197",
                            "visibility": "off"
                        }
                    },
                    {
                        "featureType": "subway",
                        "elementType": "all",
                        "stylers": {
                            "visibility": "off"
                        }
                    },
                    {
                        "featureType": "manmade",
                        "elementType": "all",
                        "stylers": {
                            "visibility": "off"
                        }
                    },
                    {
                        "featureType": "local",
                        "elementType": "all",
                        "stylers": {
                            "visibility": "off"
                        }
                    },
                    {
                        "featureType": "arterial",
                        "elementType": "labels",
                        "stylers": {
                            "visibility": "off"
                        }
                    },
                    {
                        "featureType": "boundary",
                        "elementType": "geometry.fill",
                        "stylers": {
                            "color": "#029fd4"
                        }
                    },
                    {
                        "featureType": "building",
                        "elementType": "all",
                        "stylers": {
                            "color": "#1a5787"
                        }
                    },
                    {
                        "featureType": "label",
                        "elementType": "all",
                        "stylers": {
                            "visibility": "off"
                        }
                    }
                ]
            }
        },
        series : [
            {
                type: 'custom',
                coordinateSystem: 'bmap',
                renderItem: renderItem,
                data: resData,
                itemStyle: {
                    normal: {
                        opacity: 1 //图形透明度
                    }
                },
                animation: false,
                silent: false,
            }
        ]
    };

    myChart.setOption(option);
    window.addEventListener("resize",function(){
        myChart.resize();
    });

    // 处理点击事件
    myChart.on('click', function (params) {
        // 修改barbox内容
        pointID = params.dataIndex;
        pointName = params.data.name
        document.getElementById('points').value = pointID;
        document.getElementById('uv_name').innerText = pointName;
        document.getElementById('line_graph').innerText = 'Groundtruth and Prediction (' + pointName + ')'

        /*右上折线图*/
        Newoption = createoption(data,pointID,StartInd,EndInd,MethodID);
        drawline(Newoption);

        /*error折线图*/
        RMSE_option = createRMSEoption(MethodID,pointID);
        drawRMSEhistogram(RMSE_option);

    });

}
