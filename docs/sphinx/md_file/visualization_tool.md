# 5.visualization-tool

We have developed a tool that integrates visualization, error localization, and error diagnosis. Specifically, it allows data to be uploaded, and provides interactive visual charts to show model errors, combined with spatiotemporal knowledge for error diagnosis.Welcome to visit the [website](http://39.107.116.221/) for a trial.

## 5.1. Quick Start

### 5.1.1. Quick start with predefined dataset

You can click on the dropdown menu in the 'predefined' module of the 'Data Loader', select the dataset you need, and click 'confirm' to obtain the required diagnosis and visualization.

![img](https://cdn.nlark.com/yuque/0/2023/png/25407596/1690510153791-46db1f3f-992b-411e-839f-f6aeafbf165b.png)

### 5.1.2. Quick start with prediction and ground truth

You can upload specifically formatted TSV files for prediction and ground truth in the 'upload' module of the 'Data Loader'. Clicking 'confirm' will enable you to obtain corresponding diagnosis and visualization.

![img](https://cdn.nlark.com/yuque/0/2023/png/25407596/1690511650966-1c88d248-dba3-44db-bceb-c2e052effbf9.png)

### 5.1.3. Quick start with prediction, ground truth and spatial information 

You can upload specifically formatted TSV files for prediction, ground truth and spatial information in the 'upload' module of the 'Data Loader'. Clicking 'confirm' will enable you to obtain corresponding diagnosis and visualization.

![img](https://cdn.nlark.com/yuque/0/2023/png/25407596/1690511935384-dff39c5c-3411-4973-82f0-17f554d6c28b.png)

### 5.1.4. Quick start with prediction, ground truth and temporal information 

You can upload specifically formatted TSV files for prediction, ground truth in the 'upload' module of the 'Data Loader',  along with the corresponding temporal information. Clicking 'confirm' will enable you to obtain corresponding diagnosis and visualization.

![img](https://cdn.nlark.com/yuque/0/2023/png/25407596/1690521987144-92ed25fb-80a8-46ee-8cb4-5e9c2a07040c.png)

### 5.1.5. Quick start with prediction, ground truth as well as spatial and temporal information 

You can upload specifically formatted TSV files for prediction, ground truth and spatial information in the 'upload' module of the 'Data Loader',  along with the corresponding temporal information. Clicking 'confirm' will enable you to obtain corresponding diagnosis and visualization.

![.img](https://cdn.nlark.com/yuque/0/2023/png/25407596/1690522114029-0a547ce9-f4f8-4cec-98ad-757492877022.png)

## 5.2. Contribute to our project.

The visualization-tool offers two usage options, which are accessing through the [website](http://39.107.116.221/), or using the source code(for contribution).

**Step 1: Requirements**

```vue
node == 16.14.0
npm == 8.3.1
```

**Step 2: Clone repository and install dependencies**

```Vue
git clone https://github.com/uctb/visualization-tool-UCTB.git 
cd visualization-tool-UCTB 
npm install
```

**Step 3: Start**

```Vue
npm run serve
```

You can customize the visualization tool in the source code to achieve visual effects that better fit the objectives.For better assisting you in achieving personalization of the visualization tool, we recommend following these steps to implement it.

**Step 1: Create your own component**

```vue
<template>
<div>Your own HTML</div>
</template>

<script>
export default {
	name: 'your own component name',
    data(){}
}
</script>

<style scoped>
 /*Your own CSS*/
</style>
```

**Step 2: Importing component in App.vue**

```vue
<script>
import YourOwnComponent from "./components/YourOwnComponent.vue"
export default {
	name: 'App',
    components: {
     YourOwnComponent
    }
}
</script>
```

More instructions on usage of `Vue` can be refered to on the [website](https://v2.vuejs.org/). **If you have any interesting or novel ideas, we highly welcome your pull request:)**