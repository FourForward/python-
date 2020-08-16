python-elasticsearch从创建索引到写入数据

diyiday 2018-08-28 20:07:19  11082  收藏 9
分类专栏： python
版权
创建索引

```python
from elasticsearch import Elasticsearch
es = Elasticsearch('192.168.1.1:9200')

mappings = {
            "mappings": {
                "properties": {
                    "id": {
                        "type": "long",
                        "index": "false"
                    },
                    "serial": {
                        "type": "keyword",  # keyword不会进行分词,text会分词,要使用倒排索引，需要建立索引
                        "index": "false"  # 不建索引
                    },
                    #tags可以存json格式，访问tags.content
                    "tags": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "keyword", "index": True},
                            "dominant_color_name": {"type": "keyword", "index": True},
                            "skill": {"type": "keyword", "index": True},
                        }
                    },
                    "hasTag": {
                        "type": "long",
                        "index": True
                    },
                    "status": {
                        "type": "long",
                        "index": True
                    },
                    "createTime": {
                        "type": "date",
                        "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"
                    },
                    "updateTime": {
                        "type": "date",
                        "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"
                    }
                }
            }
		}


res = es.indices.create(index = 'index_test',body =mappings)
```



通过以上代码即可创建es索引

写入一条数据
写入数据需要根据 创建的es索引类型对应的数据结构写入：

```python
from elasticsearch import Elasticsearch

es = Elasticsearch('192.168.1.1:9200')

action ={
              "id": "1111122222",
              "serial":"版本",
              #以下tags.content是错误的写法
              #"tags.content" :"标签2",
              #"tags.dominant_color_name": "域名的颜色黄色",
              #正确的写法如下：
              "tags":{"content":"标签3","dominant_color_name": "域名的颜色黄色"},
              #按照字典的格式写入，如果用上面的那种写法，会直接写成一个tags.content字段。
              #而不是在tags中content添加数据，这点需要注意
              "tags.skill":"分类信息",
              "hasTag":"123",
              "status":"11",
              "createTime" :"2018-2-2",
              "updateTime":"2018-2-3",
                }
es.index(index="index_test",doc_type="doc_type_test",body = action)
```


即可写入一条数据

写入多条数据

```python
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

es = Elasticsearch('192.168.1.1:9200')

ACTIONS = []

action1 ={
                    "_index": "indes_test",
                    "_type": "doc_type_test",
                    "_id":"bSlegGUBmJ2C8ZCSC1R1",
                    "_source":{
                        "id": "1111122222",
                        "serial":"版本",
                        "tags.content" :"标签2",
                        "tags.dominant_color_name": "域名的颜色黄色",
                        "tags.skill":"分类信息",
                        "hasTag":"123",
                        "status":"11",
                        "createTime" :"2018-2-2",
                        "updateTime":"2018-2-3",
                    }
                }
action2 ={
                    "_index": "indes_test",
                    "_type": "doc_type_test",
                    "_id":"bSlegGUBmJ2C8ZCSC1R2",
                    "_source":{
                        "id": "1111122222",
                        "serial":"版本",
                        "tags.content" :"标签2",
                        "tags.dominant_color_name": "域名的颜色黄色",
                        "tags.skill":"分类信息",
                        "hasTag":"123",
                        "status":"11",
                        "createTime" :"2018-2-2",
                        "updateTime":"2018-2-3",
                    }
                }

ACTIONS.append(action1)
ACTIONS.append(action2)

res,_ =bulk(es, ACTIONS, index="indes_test", raise_on_error=True)
print(res)
```



这个方式是手动指定了id，如果把”_id”这个参数去掉即可自动生成id数据.
如下：

    action2 ={
        "_index": "indes_test",
        "_type": "doc_type_test",
        "_source":{
            "id": "1111122222",
            "serial":"版本",
            "tags.content" :"标签2",
            "tags.dominant_color_name": "域名的颜色黄色",
            "tags.skill":"分类信息",
            "hasTag":"123",
            "status":"11",
            "createTime" :"2018-2-2",
            "updateTime":"2018-2-3",
        }
        }

#### 删除一条数据
from elasticsearch import Elasticsearch

es = Elasticsearch('192.168.1.1:9200')

res = es.delete(index="index_test",doc_type="doc_type_test", id ="bSlegGUBmJ2C8ZCSC1R1")
print(res)
直接替换id的即可删除所需的id

#### 查询一条数据
from elasticsearch import Elasticsearch

es = Elasticsearch('192.168.1.1:9200')

res = es.get(index="index_test",doc_type="doc_type_test",  id ="bSlegGUBmJ2C8ZCSC1R2")
print(res)
直接替换id的即可查询所需的id

#### 查询所有数据
from elasticsearch import Elasticsearch

es = Elasticsearch('192.168.1.1:9200')

res = es.search(index="index_test",doc_type="doc_type_test")
print(res)
print(res['hits']['hits'])

通过['hits']参数，可以解析出查询数据的详细内容

#### 根据关键词查找

from elasticsearch import Elasticsearch

es = Elasticsearch('192.168.1.1:9200')

doc = {
            "query": {
                "match": {
                    "_id": "aSlZgGUBmJ2C8ZCSPVRO"
                }
            }
        }

res = es.search(index="index_test",doc_type="doc_type_test",body=doc)
print（res）















## python 使用 elasticsearch 常用方法（检索）

```
#记录es查询等方法
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```
#清楚数据
curl -XDELETE http://xx.xx.xx.xx:9200/test6

#初始化数据
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/1' -d '{"name": "tom", "age":18, "info": "tom"}'
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/2' -d '{"name": "jack", "age":29, "info": "jack"}'
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/3' -d '{"name": "jetty", "age":18, "info": "jetty"}'
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/4' -d '{"name": "daival", "age":19, "info": "daival"}'
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/5' -d '{"name": "lilei", "age":18, "info": "lilei"}'
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/6' -d '{"name": "lili", "age":29, "info": "lili"}'
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/7' -d '{"name": "tom1", "age":30, "info": "tom1"}'
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/8' -d '{"name": "tom2", "age":31, "info": "tom2"}'
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/9' -d '{"name": "tom3", "age":32, "info": "tom3"}'
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/10' -d '{"name": "tom4", "age":33, "info": "tom4"}'
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/11' -d '{"name": "tom5", "age":34, "info": "tom5"}'
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/12' -d '{"name": "tom6", "age":35, "info": "tom6"}'
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/13' -d '{"name": "tom7", "age":36, "info": "tom7"}'
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/14' -d '{"name": "tom8", "age":37, "info": "tom8"}'
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/15' -d '{"name": "tom9", "age":38, "info": "tom9"}'
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/16' -d '{"name": "john", "age":38, "info": "john"}'
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/17' -d '{"name": "marry", "age":38, "info": "marry and john are friend"}'
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/18' -d '{"name": "john", "age":32, "info": "john"}'
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/19' -d '{"name": "tom is a little boy", "age":7, "info": "tom is a little boy"}'
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/20' -d '{"name": "tom is a student", "age":12, "info": "tom is a student"}'
curl -H "Content-Type: application/json" -XPUT 'http://xx.xx.xx.xx:9200/test6/user/21' -d '{"name": "jack is a little boy", "age":22, "info": "jack"}'
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)





```python

from elasticsearch import Elasticsearch

def foreach(doc):
    doc = res['hits']['hits']

    if len(doc):
        for item in doc:
            print(item['_source'])

es = Elasticsearch(['xx.xx.xx.xx:9200'])

#查询所有数据
#方法1
#res = es.search(index='test6', size=20)
#方法2
res = es.search(index='test6', size=20, body = {
    "query": {
        "match_all": {}
    }
})

#foreach(res)


#等于查询 term与terms, 查询 name='tom cat' 这个值不会分词必须完全包含
res = es.search(index='test6', size=20, body= {
    "query": {
        "term": {
            "name": "tom cat"
        }
    }
})
#foreach(res)

#等于查询 term与terms, 查询 name='tom' 或 name='lili'
res = es.search(index= 'test6', size= 20, body= {
    "query": {
        "terms": {
            "name": ["tom","lili"]
        }
    }
})
#foreach(res)

#包含查询，match与multi_match
# match: 匹配name包含"tom cat"关键字的数据, 会进行分词包含tom或者cat的
res = es.search(index='test6', size=20, body={
    "query": {
        "match": {
            "name": "tom cat"
        }
    }
})
#foreach(res)

#multi_match: 在name或info里匹配包含little的关键字的数据
res = es.search(index='test6', size=20, body={
    "query": {
        "multi_match": {
            "query": "little",
            "fields": ["name", "info"]
        }
    }
})
#foreach(res)


#ids , 查询id 1, 2的数据 相当于mysql的 in
res = es.search(index='test6', size=20, body={
    "query": {
        "ids": {
            "values": ["1", "2"]
        }
    }
})
#foreach(res)


#复合查询bool , bool有3类查询关系，must(都满足),should(其中一个满足),must_not(都不满足)
#name包含"tom" and term包含 "18"
res = es.search(index='test6', size=20, body={
    "query": {
        "bool": {
            "must": [
                {
                    "term": {
                        "name": "tom",
                    },

                },
                {
                    "term": {
                        "age": 18,
                    },

                },
            ]
        }
    }
})
#foreach(res)

#name包含"tom" or term包含"19"
res = es.search(index='test6', size=20, body={
    "query": {
        "bool": {
            "should": [
                {
                    "term": {
                        "name": "tom",
                    },

                },
                {
                    "term": {
                        "age": 19,
                    },

                },
            ]
        }
    }
})

#foreach(res)


#切片式查询
res = es.search(index='test6', size=20, body={
    "query": {
        "bool": {
            "should": [
                {
                    "term": {
                        "name": "tom",
                    },

                },
                {
                    "term": {
                        "age": 19,
                    },

                },
            ]
        }
    },
    "from": 2, #从第二条数据开始
    "size": 4, # 获取4条数据
})
#foreach(res)

#范围查询
res = es.search(index='test6', size=20, body={
    "query": {
        "range": {
            "age": {
                "gte": 18, #>=18
                "lte": 30  #<=30
            }
        }
    }
})
#foreach(res)


#前缀查询
res = es.search(index='test6', size=20, body={
    "query": {
        "prefix": {
            "name": "tom"
        }
    }
})
#foreach(res)


#通配符查询
res = es.search(index='test6', size=20, body={
    "query": {
        "wildcard": {
            "name": "*i"
        }
    }
})
#foreach(res)


#排序
res = es.search(index='test6', size=20, body={
    "query": {
        "wildcard": {
            "name": "*i"
        }
    },
    "sort": {
        "age": {
            "order": "desc" #降序
        }
    }
})
#foreach(res)


# count, 执行查询并获取该查询的匹配数
c = es.count(index='test6')
print(c)
# 短语匹配 match_phrase (搜索is a little的短语,不进行切分)
res = es.search(index='test6', size=20, body={
    "query": {
        "match_phrase": {
            "name": "is a little"
        }
    }
})
foreach(res)
```