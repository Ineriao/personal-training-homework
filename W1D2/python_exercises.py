class User:
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

    def describe_user(self):
        message = f"用户：{self.first_name} {self.last_name}"
        print(message)
        return message

    def greet_user(self):
        message = f"你好，{self.first_name} {self.last_name}！"
        print(message)
        return message


class Admin(User):
    def __init__(self, first_name, last_name, privileges=None):
        super().__init__(first_name, last_name)
        self.privileges = privileges or []

    def show_privileges(self):
        for privilege in self.privileges:
            print(privilege)
        return self.privileges


def max_nested_depth(values):
    if not isinstance(values, list):
        raise TypeError("values must be a list")

    child_depths = [max_nested_depth(item) for item in values if isinstance(item, list)]
    return 1 + max(child_depths, default=0)


def main():
    samples = [
        [1, 2, 3],
        [[1], [2, [3]]],
        [1, [2, [3, [4]]]],
    ]
    for sample in samples:
        print(f"{sample} 的嵌套深度为 {max_nested_depth(sample)}")

    admin = Admin(
        "张",
        "三",
        ["can add post", "can delete post", "can ban user"],
    )
    admin.describe_user()
    admin.greet_user()
    admin.show_privileges()


if __name__ == "__main__":
    main()
