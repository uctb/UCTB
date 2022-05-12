// 程序：清除按钮

    /*draw map, Groundtruth and Prediction, RMSE*/
    function drawLine() {

        /*右上折线图*/
        Initoption = createoption(data,pointID,StartInd,EndInd,MethodID);
        drawline(Initoption);

        /*error折线图*/
        FindBestMethod();
        RMSE_option = createRMSEoption(MethodID,pointID);
        drawRMSEhistogram(RMSE_option);
    }

    function drawMap() {
        console.log("now_MAPInd：", MAPInd)
        if(MAPInd == 0) {
            // Xian&Chongqing
            InitMapoption = createMapOption(data,pointID);
            drawmap(InitMapoption);
        }
        else if(MAPInd == 1) {
            // XMBaidu
            map();
        }
        else if(MAPInd == 2) {
            // NYC
        }
    }

    /* 该函数在选定数据集时执行（无论是刚读入文件，还是手动修改数据集，都会执行） */
    function FinishDataSet()
    {
        /* 确定数据集以后，先读出其中的方法数 */
        console.log("now_data:", data);
        console.log("now_DatasetList:", DatasetList);
        console.log("now_DatasetID:", DatasetID);
        console.log("now_DatasetName:", DatasetList[DatasetID]);
        FunctionNum = Object.getOwnPropertyNames(data.Pred[DatasetList[DatasetID]]).length;
        
        /* 修改数据集后，让地图点的选择先归0 */
        pointID = 0;
        document.getElementById('points').value = pointID;

        /* 读取Dataset的名称里的信息，把各项分开以后处理 */
        var dataset_name = DatasetList[DatasetID];
        var dataset_information_list = dataset_name.split('_');

        /* 获得该数据集中有效节点个数，以及相应的时间节点数 */
        var tmp = data.Pred[dataset_name];
        var TimeNum = tmp.GroundTruth.length;  // 真实值包括了多少个时间点。
        VaildPointNum = tmp.GroundTruth[0].length;
        TimeArrayLength = TimeNum;  // 标记当前选择的数据集中，每个数据点，其对应的时间节点有多少个

        console.log("now_ValidPointNum:", VaildPointNum);
        console.log("now_TimeArrayLength:", TimeArrayLength);


        /* 读取出数据集使用部分的参数，若不是all的话，进行截取 */
        if(dataset_information_list[0] != 'all') {
            portion_index = parseFloat(dataset_information_list[1]);
        }

        /* 读取最后一个参数，判断是节点型还是网格型(网格型暂时不处理) */
        datatype_index = dataset_information_list[7];
        if(datatype_index == 'G') {
            windows.alert("Grid Data is not available now.");
            return;
        }

        /* 读取小数据集的时间粒度和时间范围 */
        now_Timefitness = parseInt(dataset_information_list[6]);

        /*获得所有方法的数组*/
        var k = 0;
        var Mydata = data.Pred[dataset_name];
        for(method in Mydata){
            if(method == "GroundTruth") continue;
            else {
                MethodNameArray[k] = method;
            }
            k ++;
        }
        console.log("methodNameArray:", MethodNameArray);

        /* 修改模型误差值 */
        ChangeModelError();

        /* 获取Date类型的TimeRange，第三个元素为TimeRange的后一天 这部分好像没用了？*/
        var origin_begin = data.TimeRange[0];
        var origin_end = data.TimeRange[1];

        var origin_beginyear = Number(origin_begin.substring(0,4));
        var origin_beginmonth = Number(origin_begin.substring(5,7));
        var origin_beginday = Number(origin_begin.substring(8,10));

        var origin_endyear = Number(origin_end.substring(0,4));
        var origin_endmonth = Number(origin_end.substring(5,7));
        var origin_endday = Number(origin_end.substring(8,10));

        var Origin_beginDate = new Date(origin_beginyear,origin_beginmonth-1,origin_beginday);
        var Origin_endDate = new Date(origin_endyear,origin_endmonth-1,origin_endday);
        var EndDate = new Date(origin_endyear,origin_endmonth-1,origin_endday+1);

        TimeRange.push(Origin_beginDate)
        TimeRange.push(Origin_endDate)
        TimeRange.push(EndDate)

        /* 处理完成，作图 */
        drawMap();
        drawLine();
    }

    function ChangeModelError() {
        datasetName = DatasetList[DatasetID];
        methodName = MethodNameArray[MethodID];
        rmse = parseFloat(data['Pred'][datasetName][methodName]['rmse']).toFixed(2)
        mape = parseFloat(data['Pred'][datasetName][methodName]['mape']).toFixed(2)
        mae = parseFloat(data['Pred'][datasetName][methodName]['mae']).toFixed(2)
        document.getElementById('rmse').innerText = rmse;
        document.getElementById('mape').innerText = mape;
        document.getElementById('mae').innerText = mae;
    }

    function FinishMethod() {

        /*修改模型误差值*/
        ChangeModelError();

        /* 处理完成，作图 */
        drawMap();
        drawLine();
    }

    function ChangeDataSet(str)
    {
        DatasetID = str;
        FinishDataSet();
    }

    function ChangeMethod(str){
        MethodID = str;
        FinishMethod();
    }

    function StartDataSet()
    {
        /*增加子数据集作为options*/
        i = 0;

        for (x in DatasetList)
        {
            var dataset_information_list = DatasetList[x].split('_');
            var closeness = dataset_information_list[3];
            var period = dataset_information_list[4];
            var trend = dataset_information_list[5];
            var OptionName = 'closeness=' + closeness + ' period=' + period + ' trend=' + trend;
            document.getElementById('Dataset').options.add(new Option(OptionName, i++));
        }

        FinishDataSet();
    }

    function SetStandard()
    {
        MINACCURACY = parseFloat(document.getElementById('standard').value);

        drawMap();

        /*error折线图*/
        RMSE_option = createRMSEoption(MethodID,pointID);
        drawRMSEhistogram(RMSE_option);
    }

    function SetREStandard()
    {
        MAXError = parseFloat(document.getElementById('re_standard').value)*0.01;

        Initoption = createoption(data,pointID,StartInd,EndInd);
        drawline(Initoption);
    }

    function ClearDataSet()
    {
        DatasetList.splice(0,DatasetList.length);
        document.getElementById('Dataset').options.length = 0;
        DatasetID = 0;
    }

    /* 修改pointID/点击地图上的point时 */
    function FinishPointSelect()
    {
        pointID = parseInt(document.getElementById('points').value);

        /*作图*/
        drawMap();
        drawLine();
    }

    /*获得当前point对应的有效ID*/
    /*FinishPointSelect里用到了*/
    function FindRealNodeID(point_id)
    {
        for(x in data.Pred[DatasetList[DatasetID]])
        {
            if(x != "GroundTruth")
            {
                index_array = data.Pred[DatasetList[DatasetID]][x].traffic_data_index;
                break;
            }
        }

        return index_array.indexOf(point_id);
    }


    // 获得降序排列的各站点的RMSE
    function FindBestMethod() {

        var DatasetName = DatasetList[DatasetID];
        var Mydata = data.Pred[DatasetName];

        k = 0;

        for (method in Mydata)
        {
            if (method == "GroundTruth") continue;
            else {
                // MethodNameArray[k] = method;
                PointVar[k] = [0];

                for (i = 0; i < VaildPointNum; i++) {
                    var DictVar = {};
                    var real_node_id = Mydata[method].traffic_data_index[i]; // 注意：这里的i不等同于编号，而是对应有效点。这样才能覆盖所有有效点。

                    /* 以下方法是使用RMSE进行的判断 */
                    var total_variance = 0;
                    var total_absolute_error = 0;
                    for (j = 0; j < TimeArrayLength; j++) {
                        // k - 方法， i - 地图点， j - 时间点
                        var ground_truth = Mydata['GroundTruth'][j][i];

                        total_variance += Math.pow(Math.abs(Mydata[method].TrafficNode[j][i] - ground_truth), 2);
                        total_absolute_error += Math.abs(Mydata[method].TrafficNode[j][i] - ground_truth);
                    }

                    // 求RMSE
                    var RMSE = Math.sqrt(total_variance / TimeArrayLength);

                    // 把每个站点的index和rmse值存入字典
                    DictVar['index'] = real_node_id;
                    DictVar['rmse'] = RMSE;
                    PointVar[k][i] = DictVar;

                    if (k == 0 || PointMinVar[i] >= RMSE)      // 第一种方法，就直接存了
                    {
                        PointMinVar[i] = RMSE;
                    }
                }

                // 对RMSE数组降序排列
                PointSortedVar[k] = PointVar[k].sort((a, b) => a.rmse > b.rmse ? -1 : a.rmse < b.rmse ? 1 : 0);

                k++;
            }
        }
        console.log("sorted RMSE:", PointSortedVar);
    }


// 程序：日期栏初始化与手动设置日期（开始日期）

/* 这是给日期区域赋值的程序，读取了文件以后才会调用 */

    function ChangeFinish()
    {

        start_time = document.getElementById("starttime").value;
        end_time = document.getElementById("endtime").value;
        console.log("start_time:", start_time);

        var Year = start_time.substr(0,4);
        var Month = start_time.substr(5,2);
        var Day = start_time.substr(8,2);
        var Hour = start_time.substr(11,2);
        var Minute = start_time.substr(14,2);


        var end_Year = end_time.substr(0,4);
        var end_Month = end_time.substr(5,2);
        var end_Day = end_time.substr(8,2);
        var end_Hour = end_time.substr(11,2);
        var end_Minute = end_time.substr(14,2);

        /* 处理下这个数值 */

        const Gap = 1000*60*now_Timefitness;
        var Origin_endDate = TimeRange[2];

        var NowDate = new Date(Year,Month-1,Day,Hour,Minute);
        var NowEndDate = new Date(end_Year,end_Month-1,end_Day,end_Hour,end_Minute);

        StartInd = TimeArrayLength - ((Origin_endDate - NowDate)/ Gap);
        EndInd = TimeArrayLength - (Origin_endDate - NowEndDate)/ Gap;

        //在这里，修改起始值。另外一个就是修改结束值。
        if (StartInd > EndInd && EndInd != -1) window.alert("Error 01: Wrong time selected, please choose a correct time!");
        else
        {
            ChangeDatazoomOption = createoption(data,pointID,StartInd,EndInd);
            drawline(ChangeDatazoomOption);
        }
    }

    function writeDay(n)   //据条件写日期的下拉框   
    {   
            var e = document.reg_testdate.DD; optionsClear(e);   
            for (var i=1; i<(n+1); i++)   
                e.options.add(new Option(" "+ i + "D", i));   
    }   
    
    function IsPinYear(year)//判断是否闰平年   
    {     return(0 == year%4 && (year%100 !=0 || year%400 == 0));}   
    
    function optionsClear(e)   
    {   
        e.options.length = 1;   
    }

