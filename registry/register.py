from typing import Dict, Optional, Type, Any, Callable

class Register:
    def __init__(self,name:str):
        # 传入名字为了将来好分类
        self._name = name
        # self._register存储所有注册好的东西
        self._register:Dict[str, Any] = {}
    
    @property
    def name(self) -> str: 
        '''
        返回注册器名字
        '''
        return self._name
    
    @property
    def registry(self) -> Dict[str, Any]:
        '''
        返回已经注册的所又东西
        '''
        return self._register

    def get(self, key: str):
        '''
        返回注册到的类
        '''
        if key not in self._register:
            raise KeyError(f"Error: '{key}' is not registered in the '{self.name}' registry. "
                         f"Available keys are: {list(self._register.keys())}")

        # 返回查询到的注册到的东西
        return self._register[key]

    def register(self, name: Optional[str] = None) -> Callable:
        '''
        传入name是为了让被注册的类用名字
        '''
        def decorator(cls: Any) -> Any:
            # 注册的名字，如果没有传入名字就直接用cls的名字做，也就是允许.regster()来调用
            register_name = name if name is not None else cls.__name__.lower()
            
            # 如果已经在self._register中，则不可重复
            if register_name in self._register:
                raise ValueError(f"Error: {register_name} is already registered!")
            
            # 写到注册表中
            self._register[register_name] = cls

            return cls
        return decorator
            

# 需要啥只需要写进去就好，其实等于给注册器分类一下
MODEL = Register("model")
DATASET = Register("dataset")
LOSS = Register("loss")