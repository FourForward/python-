## Elasticsearch文档

### 1. Elasticsearch是什么

Elasticsearch是一个实时分布式搜索和分析引擎，它用于全文搜索、结构化搜索、分析以及将这三者混合使用。Elasticsearch是一个基于 [Apache Lucene(TM)](https://lucene.apache.org/core/) 的开源搜索引擎。无论在开源还是专有领域，Lucene 可以被认为是迄今为止最先进、性能最好的、功能最全的搜索引擎库。

但是，Lucene 只是一个库。想要使用它，你必须使用 Java 来作为开发语言并将其直接集成到你的应用中，更糟糕的是，Lucene 非常复杂，你需要深入了解检索的相关知识来理解它是如何工作的。

不过，Elasticsearch 不仅仅是 Lucene 和全文搜索，我们还能这样去描述它：

- 分布式的实时文件存储，每个字段都被索引并可被搜索
- 分布式的实时分析搜索引擎
- 可以扩展到上百台服务器，处理 PB 级结构化或非结构化数据

而且，所有的这些功能被集成到一个服务里面，你的应用可以通过简单的 `RESTful API`、各种语言的客户端甚至命令行与之交互。

### 2. Elasticsearch 的安装及启动

安装 Elasticsearch 之前，需要先安装一个较新的版本的 Java，最好的选择是，你可以从 https://www.oracle.com/technetwork/java/javase/downloads/index.html 获得官方提供的最新版本的 Java。

![image-20191029092135056](/Users/nfx/Library/Application Support/typora-user-images/image-20191029092135056.png)

##### 解压安装包jdk-8u231-linux-x64.tar

tar -zxvf jdk-8u231-linux-x64.tar

##### 将解压后的文件夹移到/usr/lib目录下

切换到 /usr/lib目录下

```bash
cd  /usr/lib
```

并新建jdk目录

```perl
sudo mkdir jdk
```

将解压的jdk文件复制到新建的/usr/lib/jdk目录下来

```bash
sudo mv jdk1.8.0_231 /usr/lib/jdk
```

##### 配置java环境变量

这里是将环境变量配置在 ~/.bashrc，即为所有用户配置 JDK 环境。

使用命令打开~/.bashrc文件

```shell
sudo vi ~/.bashrc
```


在末尾添加以下几行文字：

```shell
# set java env
export JAVA_HOME=/usr/lib/jdk/jdk1.8.0_231  
export PATH=$JAVA_HOME/bin:$PATH
```

执行命令使修改立即生效

```shell
source ~/.bashrc
```



![image-20191028165526423](/Users/nfx/Library/Application Support/typora-user-images/image-20191028165526423.png)

我们所使用的版本是 2.4.6，该版本较为稳定。我们首先安装 es：

```shell
tar -zxvf elasticsearch-2.4.6.tar.gz
```

解压后的目录文档如图所示：

![image-20191028101542400](/Users/nfx/Library/Application Support/typora-user-images/image-20191028101542400.png)

打开 elasticsearc-2.4.6/config/elasticsearch.yml 文件，并定位到 54 行：

![image-20191028102057816](/Users/nfx/Library/Application Support/typora-user-images/image-20191028102057816.png)

如果需要其他 ip 访问服务，则将 127.0.0.1 修改为 0.0.0.0

出于系统安全考虑的设置 ，elasticsearch不允许root用户启动，所以在启动之前需要创建一个非root用户：

```shell
创建一个新用户，用于启动elasticsearch
1）创建新用户es
	useradd es   roo
2)赋予es用户elasticsearch目录权限
	chown -R es elasticsearch目录路径
3)切换至用户es
	su es
```

配置完毕后，运行 es，执行命令 elasticsearch-2.4.6/bin/elasticsearch

![image-20191028102419522](/Users/nfx/Library/Application Support/typora-user-images/image-20191028102419522.png)

会出现 starting ... 的字样则，在浏览器出入 127.0.0.1:9200，会出现如下 json 样式：

![image-20191028102513997](/Users/nfx/Library/Application Support/typora-user-images/image-20191028102513997.png)

至此，es 服务启动完毕。

### 3. Elasticsearch原理

- 反向索引又叫倒排索引，是根据文章内容中的关键字建立索引。
- 搜索引擎原理就是建立倒排索引。
- Elasticsearch 在 Lucene 的基础上进行封装，实现了分布式搜索引擎。
- Elasticsearch 中的索引、类型和文档的概念比较重要，类似于 MySQL 中的数据库、表和行。



### 4. Elasticsearch核心的http api

#### 查看相关APIs

- ##### cat_indices

```shell
说明：
indices负责提供索引的相关信息，包括组成一个索引（index）的shard、document的数量，删除的doc数量，主存大小和所有索引的总存储大小。

命令：
GET /_cat/indices/twi*?v&s=index

返回值：
health status index    uuid                   pri rep docs.count docs.deleted store.size pri.store.size
yellow open   twitter  u8FNjxh8Rfy_awN11oDKYQ   1   1       1200            0     88.1kb         88.1kb
green  open   twitter2 nYFWZEO7TUiOjLQXBaYJpA   5   0          0            0    
```

- ##### cat aliases

- ```shell
  说明：
  aliases 负责展示当前es集群配置别名包括filter和routing信息。
  
  命令：
  GET /_cat/aliases?v
  GET /_cat/aliases/alias1,alias2
  
  返回：
  alias  index filter routing.index routing.search
  alias1 test1 -      -            -
  alias2 test1 *      -            -
  alias3 test1 -      1            1
  alias4 test1 -      2            1,2
  ```

- ##### cat allocation

```shell
说明：
allocation负责展示es的每个数据节点分配的索引分片以及使用的磁盘空间。

命令：
GET /_cat/allocation?v

返回值：
shards disk.indices disk.used disk.avail disk.total disk.percent host      ip        node
     5         260b    47.3gb     43.4gb    100.7gb           46 127.0.0.1 127.0.0.1 CSUXak2
```

- ##### cat master

```shell
说明：
master负责展示es集群的master节点信息包括节点id、节点名、ip地址等。

命令：
GET /_cat/master?v

返回值：
id                     host      ip        node
YzWoH_2BT-6UjVGDyPdqYg 127.0.0.1 127.0.0.1 YzWoH_2
```



#### 新增文档

- ##### POST方式：http://127.0.0.1:9200/spu/sku，POST方式会自动生成id

- ##### 注意：如果想新创建索引index，则url改为 http://127.0.0.1:9200/spu，类型同理。

  ##### 测试结果

  ![image-20191028162952389](/Users/nfx/Library/Application Support/typora-user-images/image-20191028162952389.png)

- ##### PUT方式：http://127.0.0.1:9200/spu/sku/2，PUT方式需要指定id，存在则更新，不存在则新增

  ##### 测试结果

  ![image-20191028163132727](/Users/nfx/Library/Application Support/typora-user-images/image-20191028163132727.png)

#### 根据主键查询

- ##### GET方式	http://127.0.0.1:9200/spu/sku/2

- ##### 返回结果中 source 表示返回的 Doucment(文档)类容，其他几个是Elasticsearch文档结构字段。如果只需要source内容不需要其他结构字段，还可以在请求url上加上属性”_source”，将只返回source部分容，请求：http://127.0.0.1:9200/spu/sku/2/source

  ##### 测试结果

  ![image-20191028163351780](/Users/nfx/Library/Application Support/typora-user-images/image-20191028163351780.png)



#### 根据主键删除

- ##### DELETE方式	http://127.0.0.1:9200/spu/sku/2

- ##### 注意：若想删除索引或者类型，只需修改 url 即可。

  ##### 测试结果

  ![image-20191028163645066](/Users/nfx/Library/Application Support/typora-user-images/image-20191028163645066.png)



#### 搜索文档

- ##### POST方式：使用match表达式进行全文搜索

- ##### 返回结果中_score是搜索引擎的概念，表示相关度，分数越高表示此文档与搜索条件关键字的匹配程度越高。

  ##### 测试结果

  ![image-20191028165238134](/Users/nfx/Library/Application Support/typora-user-images/image-20191028165238134.png)

  

### 5. 在Python中操作Elasticsearch

#### 安装 ElasticSearch 模块：

```shell
pip3 install elasticsearch==2.4.1
pip3 install elasticstack==0.4.1
pip3 install django-haystack
```



#### 基本操作：

```python
from elasticsearch import Elasticsearch

# 默认host为localhost,port为9200.但也可以指定host与port
# 在__init__方法中可以修改以下数据，根据情况自定义：
# def __init__(self, hosts=None, transport_class=Transport, **kwargs):
es = Elasticsearch()

# 1. 添加数据
# 添加或更新数据,index，doc_type名称可以自定义，id可以根据需求赋值,body为内容
es.index(index="test_index", doc_type="test_type", id=1, body={"name":"python","addr":"海淀"})
# 继续添加内容(相同index和doctype，也可不同)
es.index(index="test_index", doc_type="test_type", id=1, body={"name":"python","addr":"西二旗"})

# 2. 查询数据
# 获取索引为test_index, 文档类型为test_type的所有数据, result为一个字典类型
result = es.search(index="test_index",doc_type="test_type")
for item in result["hits"]["hits"]:
    print(item["_source"])
# 搜索id=1的文档
result = es.get(index="test_index",doc_type="test_type",id=1)
for item in result["hits"]["hits"]:
    print(item["_source"]) 

# 3. 删除数据
# 删除id=1的数据
result = es.delete(index="test_index",doc_type="test_type",id=1)
```



##### 高阶查询：

###### 1. match:匹配name包含python关键字的数据

```python
# match:匹配name包含python关键字的数据
body = {
  "query":{
    "match":{
      "name":"python"
    }
  }
}
# 查询name包含python关键字的数据
es.search(index="test_index",doc_type="test_type",body=body)
# multi_match:在name和addr里匹配包含海淀关键字的数据
body = {
  "query":{
    "multi_match":{
      "query":"海淀",
      "fields":["name","addr"]
    }
  }
}
# 查询name和addr包含"海淀"关键字的数据
es.search(index="test_index",doc_type="test_type",body=body)
```



###### 2. 搜索出id为1或2的所有数据

```python
body = {
  "query":{
    "ids":{
      "type":"test_type",
      "values":[
        "1","2"
      ]
    }
  }
}
# 搜索出id为1或2d的所有数据
es.search(index="my_index",doc_type="test_type",body=body)
```



###### 3. 从第2条数据开始，获取4条数据（切片式查询）

```python
body = {
  "query":{
    "match_all":{}
  }
  "from":2  # 从第二条数据开始
  "size":4  # 获取4条数据
}
# 从第2条数据开始，获取4条数据
es.search(index="test_index",doc_type="test_type",body=body)
```



###### 4. 查询前缀为"p"的所有数据

```python
body = {
  "query":{
    "prefix":{
      "name":"p"
    }
  }
}
# 查询前缀为"p"的所有数据
es.search(index="test_index",doc_type="test_type",body=body)
```



###### 5. 获取name="python"并且addr="海淀"的数据

```python
body = {
  "query":{
    "bool":{
      "must":[
        {
          "term":{
            "name":"python"
          }
        },
        {
          "term":{
            "addr":"海淀"
          }
        }
      ]
    }
  }
}
# 获取name="python"并且age=18的所有数据
es.search(index="test_index", doc_type="test_type", body=body)
```



###### 6. 查询name以on为后缀的所有数据

```python
body = {
  "query":{
    "wildcard":{
      "name":"*on"
    }
  }
}
# 查询name以on为后缀的所有数据
es.search(index="test_index", doc_type="test_type", body=body)
```



###### 7. 执行查询并获取该查询的匹配数

```python
# 获取数据量
es.count(index="test_index",doc_type="test_type")
```



###### 8. 搜索所有数据，并获取age最小的值以及最大的值

```python
es.index(index="test_index", doc_type="test_type", id=10, body={"name":"laowang1","age":"18"})
es.index(index="test_index", doc_type="test_type", id=11, body={"name":"laowang2","age":"28"})
es.index(index="test_index", doc_type="test_type", id=12, body={"name":"laowang3","age":"38"})

body = {
  "query":{
    "match_all":{}
  },
  "aggs":{            # 聚合查询
    "min_age":{         # 最小值的key
      "min":{         # 最小
        "field":"age"    # 查询"age"的最小值
      }
    }
  }
}
# 搜索所有数据，并获取age最小的值
es.search(index="test_index",doc_type="test_type",body=body)

body = {
  "query":{
    "match_all":{}
  },
  "aggs":{            # 聚合查询
    "max_age":{         # 最大值的key
      "max":{         # 最大
        "field":"age"    # 查询"age"的最大值
      }
    }
  }
}
# 搜索所有数据，并获取age最大的值
es.search(index="test_index",doc_type="test_type",body=body)
```



###### 9. 搜索所有数据，并获取所有age的和

```python
body = {
  "query":{
    "match_all":{}
  },
  "aggs":{            # 聚合查询
    "sum_age":{         # 和的key
      "sum":{         # 和
        "field":"age"    # 获取所有age的和
      }
    }
  }
}
# 搜索所有数据，并获取所有age的和
es.search(index="test_index",doc_type="test_type",body=body)
```

###### 更多的搜索用法：https://elasticsearch-py.readthedocs.io/en/master/api.html



### 6. Elasticsearch与Django的结合

Elasticsearch与Django的结合需要用到haystack，haystack是一款同时支持whoosh，solr，Xapian，Elasticsearch四种全文检索引擎的第三方app，我们使用的是Elasticsearch搜索引擎。

```python
pip3 install django-haystack
pip3 install elasticsearch==2.4.1
pip3 install elasticstack==0.4.1
```

##### 1. 修改settings.py

在settings.py中添加HAYSTACK的配置信息，ENGINE为使用elasticstack的引擎

```python
# Haystack
HAYSTACK_CONNECTIONS = {
    'default': {
        'ENGINE': 'elasticstack.backends.ConfigurableElasticSearchEngine',
        'URL': 'http://127.0.0.1:9200/',
        'INDEX_NAME': 'djangotest',
    },
}
# 当添加、修改、删除数据时，自动生成索引
HAYSTACK_SIGNAL_PROCESSOR = 'haystack.signals.RealtimeSignalProcessor'
# 搜索的每页大小（根据需求自行修改或添加）
HAYSTACK_SEARCH_RESULTS_PER_PAGE = 9
```

##### 2. 注册haystack的app

```
INSTALLED_APPS = [
    ...
    'haystack',
]
```

##### 3. 在goods应用目录下，添加一个索引，编辑 goods/search_indexes.py

由于达达商城的搜索功能是基于 SKU 进行检索的，所以我们需要在 goods 的 app 目录下创建 search_indexes.py 文件，且文件名称不能改变。

![image-20191028151600358](/Users/nfx/Library/Application Support/typora-user-images/image-20191028151600358.png)

如果想根据 EXAMPLE 这个 app 进行检索，则需要在 EXAMPLE 的app目录下创建 search_indexes.py 文件，特别注意文件名不能更改。

接下来我们了解一下它的哪些字段创建索引，怎么指定。

每个索引里面必须有且只能有一个字段为 document=True，这代表 haystack 和搜索引擎将使用此字段的内容作为索引进行检索(primary field)。其他的字段只是附属的属性，方便调用，并不作为检索数据。

###### 注意：如果使用一个字段设置了document=True，则一般约定此字段名为text，这是在SearchIndex类里面一贯的命名，以防止后台混乱，当然名字你也可以随便改，不过不建议改。

```python
from haystack import indexes
# 1. 改成你自己的model
from .models import SKU

# 2. 类名为模型类的名称+Index，比如模型类为Type,则这里类名为TypeIndex
class SKUIndex(indexes.SearchIndex, indexes.Indexable):
    # 指明哪些字段产生索引，产生索引的字段，会作为前端检索查询的关键字；
    # document是指明text是使用的文档格式，产生字段的内容在文档中进行描述；
    # use_template是指明在模板中被声明需要产生索引；
    text = indexes.CharField(document=True, use_template=True)

    # 3. 返回的是你自己的model
    def get_model(self):
        """返回建立索引的模型类"""
        return SKU

    # 4. 修改return 可以修改返回查询集的内容，比如返回时，有什么条件限制
    def index_queryset(self, using=None):
        """返回要建立索引的数据查询集"""
        return self.get_model().objects.filter(is_launched=True)
```

并且，haystack提供了 use_template = True 在 text 字段，这样就允许我们使用数据模板去建立搜索引擎索引的文件，说得通俗点就是索引里面需要存放一些什么东西，例如 SKU 的 name 字段，这样我们可以通过 name 内容来检索 SKU 数据了，举个例子，假如你搜索手提包 ，那么就可以检索出 name 中含有手提包的 SKU 了。

##### 4. 创建数据模版

数据模版的路径也需要注意路径，以及文件名必须为  `要索引的类名_text.txt`：

![image-20191028153101220](/Users/nfx/Library/Application Support/typora-user-images/image-20191028153101220.png)

sku_text.txt的内容如下：


![image-20191028153309241](/Users/nfx/Library/Application Support/typora-user-images/image-20191028153309241.png)

在这里我们可以通过 SKU 的id、name、caption 字段来进行搜索。

###### 注意：如果在 search_indexes.py 文件中的 text 字段的 use_template 的值设置为 false，则不能用模版进行检索。

##### 5. 修改 views.py

```python
class GoodsSearchView(View):
    def post(self,request, load_all=True, searchqueryset=None, extra_context=None):
        """
        首页查询功能
        :param request:
        :return:
        """
        # 127.0.0.1:8000/v1/goods/search/
        if request.method == "POST":
            query = ''
            page_size = 2
            results = EmptySearchQuerySet()
            if request.POST.get('q'):
                form = ModelSearchForm(request.POST, searchqueryset=searchqueryset, load_all=load_all)
                if form.is_valid():
                    query = form.cleaned_data['q']
                    results = form.search()
            else:
                form = ModelSearchForm(request.POST, searchqueryset=searchqueryset, load_all=load_all)

            paginator = Paginator(results, page_size)
            try:
                page = paginator.page(int(request.POST.get('page', 1)))
            except:
                result = {'code': 40200, 'data': '页数有误，小于0或者大于总页数'}
                return JsonResponse(result)

            # 记录查询信息
            context = {
                'form': form,
                'page': page,
                'paginator': paginator,
                'query': query,
                'suggestion': None,
            }

            if results.query.backend.include_spelling:
                context['suggestion'] = form.get_suggestion()

            if extra_context:
                context.update(extra_context)

            sku_list = []
            # print(len(page.object_list))
            for result in page.object_list:
                sku = {
                    'skuid': result.object.id,
                    'name': result.object.name,
                    'price': result.object.price,
                }
                sku_image_count = SKUImage.objects.filter(sku_id=result.object.id).count()
                # 如果该sku没有指定图片，则选取默认图片进行展示
                if sku_image_count == 0:
                    sku_image = str(result.object.default_image_url)
                else:
                    sku_image = str(SKUImage.objects.get(sku_id=result.object.id).image)
                sku['image'] = sku_image
                sku_list.append(sku)
            result = {"code": 200, "data": sku_list, 'paginator': {'pagesize': page_size, 'total': len(results)}}
            return JsonResponse(result)
```



##### 6. 生成索引

在生成索引前同步一下数据库：

```python
python3 manage.py makemigrations
python3 manage.py migrate
```

需要手动生成一次索引，之后会跟据settings.py中的配置自动生成索引：

```python
python3 manage.py rebuild_index
```



##### 7. 启动项目并测试

```python
python3 manage.py runserver 0.0.0.0:8000
```

















