import unittest

from python_exercises import Admin, User, max_nested_depth


class MaxNestedDepthTests(unittest.TestCase):
    def test_flat_list_depth_is_one(self):
        self.assertEqual(max_nested_depth([1, 2, 3]), 1)

    def test_nested_list_depth_is_three(self):
        self.assertEqual(max_nested_depth([[1], [2, [3]]]), 3)

    def test_empty_list_depth_is_one(self):
        self.assertEqual(max_nested_depth([]), 1)

    def test_mixed_list_depth_ignores_non_list_values(self):
        self.assertEqual(max_nested_depth([1, "text", [True, 2]]), 2)

    def test_deep_list_depth(self):
        self.assertEqual(max_nested_depth([[[[1]]]]), 4)

    def test_non_list_argument_raises_type_error(self):
        with self.assertRaises(TypeError):
            max_nested_depth("not a list")


class UserAndAdminTests(unittest.TestCase):
    def test_user_has_name_attributes(self):
        user = User("张", "三")
        self.assertEqual(user.first_name, "张")
        self.assertEqual(user.last_name, "三")

    def test_user_methods_return_messages(self):
        user = User("张", "三")
        self.assertEqual(user.describe_user(), "用户：张 三")
        self.assertEqual(user.greet_user(), "你好，张 三！")

    def test_admin_inherits_user(self):
        admin = Admin("李", "四", ["can add post"])
        self.assertIsInstance(admin, User)
        self.assertEqual(admin.privileges, ["can add post"])

    def test_admin_shows_privileges(self):
        privileges = ["can add post", "can delete post", "can ban user"]
        admin = Admin("王", "五", privileges)
        self.assertEqual(admin.show_privileges(), privileges)


if __name__ == "__main__":
    unittest.main()
